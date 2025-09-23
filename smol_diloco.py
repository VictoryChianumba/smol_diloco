# smol_diloco.py
import math, os, random,  argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

# --------------------
# Utilities 
# --------------------

def setup_ddp():
    # Detect if launched under torchrun (env vars present)
    has_dist_env = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Device: prefer MPS (Apple Silicon), allow forcing CPU via env
    force_cpu = os.environ.get("TORCH_DEVICE", "") == "cpu"
    use_mps = (not force_cpu) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = torch.device("mps") if use_mps else torch.device("cpu")

    # Only init process group if truly multi-process
    if has_dist_env and world_size > 1:
        dist.init_process_group(backend="gloo", init_method=os.environ.get("INIT_METHOD", "env://"))

    return rank, world_size, device


def barrier():
    if dist.is_initialized():
        dist.barrier()  
        
def all_reduce_mean(tensors: Iterable[torch.Tensor]):
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
        
def broadcast_(tensors: Iterable[torch.Tensor], src: int = 0):
    for t in tensors:
        dist.broadcast(t, src)

def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed); 
    
# ------------------------
# Tiny toy tokenizer/data
# ------------------------

class ToyCharDataset(torch.utils.data.Dataset):
    def __init__ (self, length: int=200000, ctx:int=64, vocab:int=96, seed:int=123):
        g = torch.Generator().manual_seed(seed) 
        self.data = torch.randint(low=0, high=vocab, size = (length+1,),generator = g)
        # What does this stand for?
        self.ctx = ctx
    
    def __len__(self):
        return len(self.data) - self.ctx - 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.ctx]
        y = self.data[idx+1:idx+self.ctx+1]
        return x, y
    
# ---------------------------
# A tiny token MLP "LM"
# ---------------------------

class TinyTokenMLP(nn.Module):
    def __init__(self, vocab=96, hidden= 256):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4*hidden), 
            nn.GELU(), 
            nn.Linear(4*hidden, hidden),
        )
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias = False)
    
    def forward(self, x):
        h = self.emb(x)
        h = self.ffn(h) + h
        h = self.ln(h)
        logits = self.head(h)
        return logits
    
    
# ---------------------------
# Outer optimizer (Nesterov-ish momentum) on server
# ---------------------------

@dataclass 
class OuterOptState:
    momentum: list

def nesterov_update(server_params, grad_like, state: OuterOptState, lr: float, momentum:float):
    # grad_like here is (+) average delta from workers (0_local - 0_server_start).
    with torch.no_grad():
        for p, g, v in zip(server_params, grad_like, state.momentum):
            v.mul_(momentum).add_(g)
            p.add_(v, alpha=lr*(1.0 + momentum))  
    return state

def clone_like(params: Iterable[torch.Tensor]) -> list:
    return [p.detach().clone() for p in params]

def zeros_like(params: Iterable[torch.Tensor]) -> list:
    return [torch.zeros_like(p) for p in params]

def add_inplace(dst_list, src_list, alpha=1.0):
    for d, s in zip(dst_list, src_list):
        d.add(s, alpha=alpha)
        
def sub_lists(a, b):
    return [x-y for (x, y) in zip(a, b)]

def load_params_(params:Iterable[torch.nn.Parameter], src_list:Iterable[torch.Tensor]):
    with torch.no_grad():
        for p, s in zip(params, src_list):
            p.copy_(s)
            
# --------------------------
# Training helpers 
# --------------------------

def shard_dataloader(ds, rank, world_size, batch_size):
    # simple round-robin sharding
    indices = torch.arange(rank, len(ds), world_size)
    sampler = torch.utils.data.SubsetRandomSampler(indices) 
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler = sampler, drop_last=True)

def loss_on_batch(model, batch, device):
    x, y = batch
    x = x.to(device); y = y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    return loss

# ----------------
# Main DiLoCo loop
# ---------------

def run(args):
    rank, world_size, device = setup_ddp()
    set_seed(args.seed + rank)
    
    # Model & data
    model = TinyTokenMLP(vocab=args.vocab, hidden=args.hidden).to(device)
    ds = ToyCharDataset(length=args.tokens, ctx=args.ctx, vocab = args.vocab, seed=args.seed)
    loader = shard_dataloader(ds, rank, world_size, args.batch_size)    
    
    # Inner optimizer (per DiLoCo spec: AdamW commonly used)
    inner_opt = torch.optim.AdamW(model.parameters(), lr = args.inner_lr, betas=(0.9, 0.95), weight_decay = args.weight_decay)
    
    # server params, buffers, and outer state (kept on all ranks; rank0 is authoritative)
    server_params = clone_like([p.data for p in model.parameters()])
    outer_state = OuterOptState(momentum=zeros_like(server_params))
    # Broadcast initial server weights
    for buf in server_params: buf.data = buf.to(device) 
    if world_size > 1 and dist.is_initialized():
        # rank0 holds truth; broadcast server params
        if rank == 0:
            tensors = [b.data for b in server_params]
        else:
            tensors = [torch.empty_like(b) for b in server_params]
        broadcast_(tensors, src=0)
        load_params_(server_params, tensors)
    load_params_(model.parameters(), server_params)
    
    # Training 
    it = iter(loader)
    for round_idx in range(args.rounds):
        # Reset inner optmizer state each round (common/simple choice)
        inner_opt = torch.optim.AdamW(model.parameters(), lr = args.inner_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        
        # Snapshot of starting server weights for delta computation 
        start_params = clone_like(server_params)
        
        model.train()
        # ------------- local steps (no cross-node communication) ----
        for step in range(args.local_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch  = next(it)
                
            loss = loss_on_batch(model, batch, device)
            inner_opt.zero_grad(set_to_none = True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            inner_opt.step()    
            
            if step % max(1, args.log_every) == 0 and rank == 0:
                print(f"[round {round_idx:03d} | step {step:03d}] loss={loss.item():.3f}", flush=True)
                # dfjkdsla;jf;dlsaf
                # If you want retype from here. From here forward is pasted
                
                # jjjfjfjjfjfjfjf
        
        # ---- (A) Save server norm BEFORE outer update
        if rank == 0:
            with torch.no_grad():
                server_norm_before = torch.sqrt(sum((p.data.float()**2).sum() for p in server_params))

        # ---- (B) Compute local delta (θ_i^K − θ_server_start)
        with torch.no_grad():
            local_params = [p.data for p in model.parameters()]
            local_delta  = sub_lists(local_params, start_params)

        # ---- (C) Average deltas across workers
        all_reduce_mean(local_delta)  # your helper; no-op single-process

        # ---- (D) Log global delta size (diagnostic)
        if rank == 0:
            with torch.no_grad():
                delta_norm = torch.sqrt(sum((d.float()**2).sum() for d in local_delta))
                print(f"[round {round_idx:03d}] avg_delta_norm={delta_norm.item():.3e}")

        # ---- (E) Apply outer update
        if rank == 0:
            nesterov_update(server_params, local_delta, outer_state,
                            lr=args.outer_lr, momentum=args.outer_momentum)

        barrier()

        # ---- (F) Broadcast updated server params; load into model (unchanged)
        tensors = [p.data for p in server_params]
        broadcast_(tensors, src=0)
        load_params_(model.parameters(), tensors)

        # ---- (G) Now measure the ACTUAL server step size (AFTER update)
        if rank == 0:
            with torch.no_grad():
                step_norm = torch.sqrt(sum(((p.data - s).float()**2).sum()
                                        for p, s in zip(server_params, start_params)))
                print(f"[round {round_idx:03d}] server_norm_before={server_norm_before.item():.3e} "
                    f"step_norm={step_norm.item():.3e}")


        # Optional: small eval on a mini-batch (here just report loss on one batch)
        if args.eval_every > 0 and (round_idx + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                try:
                    eb = next(it)
                except StopIteration:
                    it = iter(loader); eb = next(it)
                eloss = loss_on_batch(model, eb, device)
                
        if args.eval_every > 0 and (round_idx + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                try:
                    eb = next(it)
                except StopIteration:
                    it = iter(loader); eb = next(it)
                eloss = loss_on_batch(model, eb, device)
            if rank == 0:
                print(f"[round {round_idx:03d}] eval_loss={eloss.item():.3f}", flush=True)

    # Save checkpoint from rank0
    if rank == 0 and args.ckpt:
        torch.save({"model": [p.cpu() for p in model.parameters()]}, args.ckpt)
        print(f"Saved {args.ckpt}")

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200_000, help="synthetic tokens for the toy dataset")
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--local_steps", type=int, default=50, help="K inner AdamW steps per round")
    ap.add_argument("--rounds", type=int, default=50, help="number of outer rounds")
    ap.add_argument("--inner_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--outer_lr", type=float, default=0.1, help="server step size on averaged delta")
    ap.add_argument("--outer_momentum", type=float, default=0.9, help="Nesterov momentum coefficient")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ckpt", type=str, default="smol_diloco.pt")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
                
                