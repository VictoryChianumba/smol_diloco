# smol_diloco.py
import math, os, random, json, time, argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from datasets import ToyCharDataset, TinyShakespeareDataset, SeqDataset, speaker_shard_ids
from models import TinyTokenMLP, CausalSelfAttention, TinyTransformerLM, TransformerBlock
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
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def all_reduce_mean_(tensors):
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()

def broadcast_(tensors, src=0):
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return
    for t in tensors:
        dist.broadcast(t, src)


def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed);


# --------------------
# Metric logging
# --------------------

class MetricLogger:
    """Append per-round metrics to <log_dir>/metrics.jsonl and dump config.json.

    Only rank 0 writes. A no-op when log_dir is empty so the training loop
    stays unchanged for ad-hoc runs.
    """
    def __init__(self, log_dir: str, rank: int, config: dict):
        self.enabled = bool(log_dir) and rank == 0
        if not self.enabled:
            return
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "metrics.jsonl")
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        # truncate any previous run in this dir
        open(self.path, "w").close()

    def log(self, record: dict):
        if not self.enabled:
            return
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")



  
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
        d.add_(s, alpha=alpha)
        
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

@torch.no_grad()
def sample_text(model, itos, ctx, seed_token=None, steps=200, temperature=1.0, top_k=50, device="cpu"):
    model.eval()
    # start from a single token; default to ' ' (space) if present, else 0
    if seed_token is None:
        seed_token = 0
        if ' ' in itos.values():
            # find the index of space if available
            for i, ch in itos.items():
                if ch == ' ':
                    seed_token = i
                    break
    x = torch.tensor([[seed_token]], dtype=torch.long, device=device)
    out_chars = []
    for _ in range(steps):
        x_ctx = x[:, -ctx:]
        logits = model(x_ctx)[:, -1, :]
        logits = logits / max(1e-8, temperature)
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k=k)
            thresh = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
        x = torch.cat([x, next_id], dim=1)
        out_chars.append(itos[int(next_id.item())])
    return "".join(out_chars)


# ----------------
# Main DiLoCo loop
# ---------------

def run(args):
    rank, world_size, device = setup_ddp()
    set_seed(args.seed + rank)
    
    # === Dataset (+ train/val split) ===
    use_tiny = args.dataset == "tinyshakespeare"

    if use_tiny:
        ds_full = TinyShakespeareDataset(ctx=args.ctx, path=getattr(args, "data_path", "tiny_shakespeare.txt"))
        vocab_size = ds_full.vocab_size

        if args.shard == "non_iid":
            # Split the raw text 95/5 first, then partition the train side by
            # speaker so workers see disjoint characters. The val set is a
            # held-out region of the corpus -- no worker trained on it.
            text = ds_full.text
            text_split = int(0.95 * len(text))
            train_text, val_text = text[:text_split], text[text_split:]
            per_rank = speaker_shard_ids(train_text, ds_full.stoi, world_size)
            ds_train = SeqDataset(per_rank[rank], ctx=args.ctx)
            val_ids = torch.tensor([ds_full.stoi[c] for c in val_text if c in ds_full.stoi], dtype=torch.long)
            ds_val = SeqDataset(val_ids, ctx=args.ctx)
        else:
            N = len(ds_full.data)
            split = int(0.95 * N)
            train_ids = ds_full.data[:split]
            val_ids   = ds_full.data[split:]
            ds_train = SeqDataset(train_ids, ctx=args.ctx)
            ds_val   = SeqDataset(val_ids,   ctx=args.ctx)
    else:
        ds_train = ToyCharDataset(length=args.tokens, ctx=args.ctx, vocab=args.vocab, seed=args.seed)
        ds_val   = ToyCharDataset(length=max(10_000, int(args.tokens*0.05)), ctx=args.ctx, vocab=args.vocab, seed=args.seed+1)
        vocab_size = args.vocab

    # Training loader: in non-IID mode every rank already holds its own dataset,
    # so just iterate it; in IID mode shard a single dataset round-robin. Val is
    # always unsharded and only rank 0 evaluates it.
    if args.shard == "non_iid" and use_tiny:
        loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        loader = shard_dataloader(ds_train, rank, world_size, args.batch_size)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=True)


    # Model (use dataset-derived vocab)
    if args.model == "transformer":
        model = TinyTransformerLM(vocab=vocab_size, ctx=args.ctx, n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer).to(device)
    else:
        model = TinyTokenMLP(vocab=vocab_size, hidden=args.hidden).to(device)   
        
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

    # === Metric logging setup ===
    n_params = sum(p.numel() for p in model.parameters())
    # Bytes moved across the network per sync round. Each round every worker
    # contributes its delta to an all-reduce and receives the averaged result;
    # a ring/recursive all-reduce moves ~2*(W-1)/W * payload per worker. We then
    # broadcast the updated server params (~1 payload). fp32 => 4 bytes/param.
    payload = n_params * 4
    bytes_per_round = (2.0 * (world_size - 1) / max(1, world_size) + 1.0) * payload if world_size > 1 else 0.0
    logger = MetricLogger(args.log_dir, rank, {
        **vars(args),
        "world_size": world_size,
        "n_params": n_params,
        "vocab_size": vocab_size,
        "bytes_per_round": bytes_per_round,
    })
    cum_inner_steps = 0
    cum_comm_bytes = 0.0

    # Inner optimizer. The DiLoCo paper resets it at every outer step (K~500, so
    # the reset is amortized). For a sync-interval sweep that reaches small K,
    # resetting AdamW each round would stop it ever accumulating second-moment
    # estimates -- an optimizer artifact that confounds the communication study.
    # So we persist it by default; --reset_inner_opt restores the paper's behavior.
    inner_opt = torch.optim.AdamW(model.parameters(), lr=args.inner_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # Training
    it = iter(loader)
    for round_idx in range(args.rounds):
        round_t0 = time.time()
        if args.reset_inner_opt:
            inner_opt = torch.optim.AdamW(model.parameters(), lr=args.inner_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

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
            last_loss = loss.item()

            if step % max(1, args.log_every) == 0 and rank == 0:
                print(f"[round {round_idx:03d} | step {step:03d}] loss={last_loss:.3f}", flush=True)
        cum_inner_steps += args.local_steps
              
        # save server norm BEFORE outer update
        if rank == 0:
            with torch.no_grad():
                server_norm_before = torch.sqrt(sum((p.data.float()**2).sum() for p in server_params))

        # compute local delta (θ_i^K − θ_server_start)
        with torch.no_grad():
            local_params = [p.data for p in model.parameters()]
            local_delta  = sub_lists(local_params, start_params)

        # Average deltas across workers
        all_reduce_mean_(local_delta)  # your helper; no-op single-process

        cum_comm_bytes += bytes_per_round

        # global delta size (diagnostic)
        delta_norm_val = None
        if rank == 0:
            with torch.no_grad():
                delta_norm = torch.sqrt(sum((d.float()**2).sum() for d in local_delta))
                delta_norm_val = delta_norm.item()
                print(f"[round {round_idx:03d}] avg_delta_norm={delta_norm_val:.3e}")

        # outer update
        if rank == 0:
            nesterov_update(server_params, local_delta, outer_state,
                            lr=args.outer_lr, momentum=args.outer_momentum)

        barrier()

        # Broadcast updated server params; load into model (unchanged)
        tensors = [p.data for p in server_params]
        broadcast_(tensors, src=0)
        load_params_(model.parameters(), tensors)

        # server step size (AFTER update)
        step_norm_val = None
        if rank == 0:
            with torch.no_grad():
                step_norm = torch.sqrt(sum(((p.data - s).float()**2).sum()
                                        for p, s in zip(server_params, start_params)))
                step_norm_val = step_norm.item()
                print(f"[round {round_idx:03d}] server_norm_before={server_norm_before.item():.3e} "
                    f"step_norm={step_norm_val:.3e}")


        # === Validation (rank 0 only) + sample text ===
        # --val_batches caps the eval cost so it doesn't dominate wall-clock and
        # so timing is comparable across runs; 0 means iterate the full split.
        val_loss = None
        if args.eval_every > 0 and (round_idx + 1) % args.eval_every == 0 and rank == 0:
            model.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for vb in val_loader:
                    total += loss_on_batch(model, vb, device).item()
                    count += 1
                    if args.val_batches and count >= args.val_batches:
                        break
            val_loss = total / max(1, count)
            print(f"[round {round_idx:03d}] val_loss={val_loss:.3f}", flush=True)

            # Optional: print a short sample (works when TinyShakespeareDataset is used)
            if args.sample and 'ds_full' in locals() and hasattr(ds_full, "itos"):
                txt = sample_text(model, ds_full.itos, ctx=args.ctx, steps=200, temperature=1.0, top_k=50, device=device)
                print("-" * 60)
                print(txt)
                print("-" * 60)

        # === Persist this round's metrics ===
        if rank == 0:
            tokens_seen = cum_inner_steps * args.batch_size * args.ctx * max(1, world_size)
            logger.log({
                "round": round_idx,
                "inner_steps": cum_inner_steps,
                "tokens_seen": tokens_seen,
                "wall_time_s": round(time.time() - round_t0, 4),
                "train_loss": last_loss,
                "val_loss": val_loss,
                "avg_delta_norm": delta_norm_val,
                "server_step_norm": step_norm_val,
                "comm_bytes": cum_comm_bytes,
            })


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
    ap.add_argument("--reset_inner_opt", action="store_true", help="reset inner AdamW state each round (DiLoCo paper behavior)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--outer_lr", type=float, default=0.1, help="server step size on averaged delta")
    ap.add_argument("--outer_momentum", type=float, default=0.9, help="Nesterov momentum coefficient")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--val_batches", type=int, default=0, help="cap eval to this many batches (0 = full val split)")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--log_dir", type=str, default="", help="if set, write metrics.jsonl + config.json here")
    ap.add_argument("--sample", action="store_true", help="print a text sample at each eval")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ckpt", type=str, default="smol_diloco.pt")
    ap.add_argument("--dataset", type=str, default="tinyshakespeare", choices=["tinyshakespeare", "toy"])
    ap.add_argument("--data_path", type=str, default="tiny_shakespeare.txt")
    ap.add_argument("--shard", type=str, default="iid", choices=["iid", "non_iid"],
                    help="iid: round-robin sharding of one corpus; non_iid: each worker sees a disjoint set of Shakespeare speakers")
    ap.add_argument("--model", type=str, default="transformer", choices=["transformer", "mlp"])
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=2)
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
                
                