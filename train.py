"""
Autoresearch pretraining script. Single-GPU, single-file.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import inspect
import math
import subprocess
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, EVAL_TOKENS, Tokenizer, make_dataloader


# ---------------------------------------------------------------------------
# Muon Optimizer (Newton-Schulz orthogonalization for 2D params)
# ---------------------------------------------------------------------------

def newton_schulz_(G, steps=5):
    """In-place Newton-Schulz orthogonalization of a matrix."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

class Muon(torch.optim.Optimizer):
    """Muon optimizer: Newton-Schulz orthogonalization for 2D weight matrices."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if g.ndim == 2:
                    g = newton_schulz_(g, steps=ns_steps)
                p.add_(g, alpha=-lr)

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 1024      # maximum sequence length (context length)
    vocab_size: int = 50304       # number of tokens
    n_layer: int = 20             # number of transformer blocks
    n_head: int = 8               # number of attention heads
    n_embd: int = 512             # embedding dimension
    use_rope: bool = True         # use RoPE positional encoding
    rope_base: float = 10000.0
    mlp_type: str = "swiglu"      # MLP activation: "gelu" or "swiglu"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Gradient accumulation
TOTAL_BATCH_SIZE = 65536        # 2**16, ~64K tokens per optimizer step
DEVICE_BATCH_SIZE = 16          # per-device batch size (fits RTX 3090 24GB)

# Learning rate schedule (cosine decay with warmup)
MAX_LR = 1.8e-3                # 1.8e-3 (for AdamW on embeddings)
MIN_LR = MAX_LR * 0.1          # min LR
MUON_LR = 0.015                # Muon LR for 2D transformer weights
WARMUP_STEPS = 50

# Optimizer
WEIGHT_DECAY = 0.1
GRAD_CLIP = 0.5                 # global gradient norm clipping

# GPU Performance (RTX 3090) - Do not change this. It is static.
GPU_BF16_PEAK_FLOPS = 71.2e12   # RTX 3090 BF16 tensor core TFLOPS (FP32 accumulate, dense)



class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, max_position_embeddings=1024):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.size(-2)
        cos = self.cos_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[..., :seq_len, :].to(dtype=q.dtype, device=q.device)
        q2 = (q * cos) + (self._rotate_half(q) * sin)
        k2 = (k * cos) + (self._rotate_half(k) * sin)
        return q2, k2


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.use_rope = getattr(config, "use_rope", True)
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                base=getattr(config, "rope_base", 10000.0),
                max_position_embeddings=config.sequence_len,
            )
        else:
            self.rotary_emb = None

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, seq_len=T)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_type = getattr(config, "mlp_type", "gelu")
        self.mlp_type = mlp_type
        if mlp_type == "swiglu":
            inner_dim = int(4 * config.n_embd * 2 / 3)
            inner_dim = ((inner_dim + 255) // 256) * 256
            self.inner_dim = inner_dim
            self.c_fc = nn.Linear(config.n_embd, 2 * inner_dim)
            self.c_proj = nn.Linear(inner_dim, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.gelu = nn.GELU(approximate='tanh')
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        if self.mlp_type == "swiglu":
            x_in = self.c_fc(x)
            x_gate, x_up = x_in.chunk(2, dim=-1)
            x = F.silu(x_gate) * x_up
            x = self.c_proj(x)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            dpn=nn.RMSNorm(config.n_embd),
            ln_f=nn.RMSNorm(config.n_embd),
        ))
        if not getattr(config, "use_rope", True):
            self.transformer["wpe"] = nn.Embedding(config.sequence_len, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def init_weights(self):
        self.apply(self._init_weights)

    def num_scaling_params(self):
        wte = self.transformer.wte.weight.numel()
        transformer_h = sum(p.numel() for p in self.transformer.h.parameters())
        h_ids = {id(p) for p in self.transformer.h.parameters()}
        wte_id = id(self.transformer.wte.weight)
        other = sum(
            p.numel() for p in self.parameters()
            if id(p) not in h_ids and id(p) != wte_id
        )
        total = wte + transformer_h + other
        return {'wte': wte, 'transformer_matrices': transformer_h, 'other': other, 'total': total}

    def estimate_flops(self):
        seen = set()
        nparams = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                nparams += p.numel()
        nparams_exclude = self.transformer.wte.weight.numel()
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * q * t
        return 6 * (nparams - nparams_exclude) + attn_flops

    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose=True):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Split: 2D transformer weights -> Muon, embeddings + 1D -> AdamW
        muon_params = []
        adam_decay_params = []
        adam_nodecay_params = []
        for name, p in param_dict.items():
            if p.dim() >= 2 and 'wte' not in name and 'lm_head' not in name:
                muon_params.append(p)
            elif p.dim() >= 2:
                adam_decay_params.append(p)
            else:
                adam_nodecay_params.append(p)

        if verbose:
            print(f"Muon params: {len(muon_params)}, {sum(p.numel() for p in muon_params):,}")
            print(f"AdamW decay params: {len(adam_decay_params)}, {sum(p.numel() for p in adam_decay_params):,}")
            print(f"AdamW nodecay params: {len(adam_nodecay_params)}, {sum(p.numel() for p in adam_nodecay_params):,}")

        muon_optimizer = Muon(muon_params, lr=MUON_LR, momentum=0.95)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        adam_optimizer = torch.optim.AdamW(
            [{'params': adam_decay_params, 'weight_decay': weight_decay},
             {'params': adam_nodecay_params, 'weight_decay': 0.0}],
            lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return muon_optimizer, adam_optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        _, T = idx.size()
        assert T <= self.config.sequence_len, \
            f"Cannot forward sequence of length {T}, sequence_len is only {self.config.sequence_len}"
        tok_emb = self.transformer.wte(idx)
        if getattr(self.config, "use_rope", True):
            x = tok_emb
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = tok_emb + pos_emb
        x = self.transformer.dpn(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits



def get_gpu_stats():
    """Query nvidia-smi for temperature (C) and power draw (W)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=2,
        ).decode().strip()
        temp_str, power_str = out.split(",")
        return int(temp_str.strip()), float(power_str.strip())
    except Exception:
        return 0, 0.0


@torch.no_grad()
def evaluate_val_loss(model, tokenizer, batch_size):
    """Standard cross-entropy validation loss averaged over EVAL_TOKENS."""
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_loss = 0.0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss = model(x, y, reduction='mean')
        total_loss += loss.item()
    return total_loss / steps


# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = GPTConfig(sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size)
print(f"Model config: {asdict(config)}")

model = GPT(config).to(device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

muon_optimizer, adam_optimizer = model.configure_optimizers(
    weight_decay=WEIGHT_DECAY,
    learning_rate=MAX_LR,
    device_type="cuda",
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Cosine decay LR schedule with linear warmup
# We estimate max_steps from the time budget after the first step
estimated_max_steps = 1000  # initial estimate, updated after first step

DECAY_FRACTION = 0.15  # fraction of training spent in decay phase

def get_lr(step):
    # Linear warmup
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    # WSD: stable at MAX_LR, then linear decay in last DECAY_FRACTION of training
    decay_start = int(estimated_max_steps * (1 - DECAY_FRACTION))
    if step < decay_start:
        return MAX_LR
    if step > estimated_max_steps:
        return MIN_LR
    # Cosine decay from MAX_LR to MIN_LR
    decay_ratio = (step - decay_start) / (estimated_max_steps - decay_start)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0
first_step_dt = 0.0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    # --- Gradient accumulation ---
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        x, y, epoch = next(train_loader)

    # Gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    # Set learning rate for this step
    lr = get_lr(step)
    muon_lr_scale = lr / MAX_LR  # scale Muon LR proportionally
    for param_group in adam_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in muon_optimizer.param_groups:
        param_group['lr'] = MUON_LR * muon_lr_scale

    muon_optimizer.step()
    adam_optimizer.step()
    muon_optimizer.zero_grad(set_to_none=True)
    adam_optimizer.zero_grad(set_to_none=True)

    train_loss_f = loss_accum.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    total_training_time += dt

    # After 5 steps, estimate max_steps for LR schedule using last 4 steps
    # (step 0 includes torch.compile overhead and is not representative)
    if step == 4:
        avg_dt = (total_training_time - first_step_dt) / 4
        estimated_max_steps = max(int(TIME_BUDGET / avg_dt), WARMUP_STEPS + 10)
    if step == 0:
        first_step_dt = dt

    # Logging
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / GPU_BF16_PEAK_FLOPS
    remaining = TIME_BUDGET - total_training_time

    gpu_temp, gpu_power = get_gpu_stats()
    gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"step {step:05d} | loss: {train_loss_f:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s | {gpu_temp}°C {gpu_power:.0f}W {gpu_mem_gb:.1f}GB", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Stop when time budget is exhausted
    if total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_loss = evaluate_val_loss(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()

print("---")
print(f"val_loss:         {val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"mfu_percent:      {mfu:.2f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
