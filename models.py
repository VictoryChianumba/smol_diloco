# ---------------------------
# Tiny Transformer (causal LM)
# ---------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, ctx):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        # register causal mask buffer
        mask = torch.tril(torch.ones(ctx, ctx, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)                      # [B,T,3C]
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,h,T,Hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)      # [B,h,T,T]
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = att.softmax(dim=-1)
        y = att @ v                                                   # [B,h,T,Hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, ctx, mlp_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, ctx)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_mult * n_embd),
            nn.GELU(),
            nn.Linear(mlp_mult * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab, ctx, n_embd=256, n_head=4, n_layer=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(ctx, n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, ctx) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.ctx = ctx

    def forward(self, x):  # x: [B,T]
        B, T = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)  # [1,T]
        h = self.emb(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)

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
    