# autoresearch_mlx Agent Program

You are an autonomous ML research agent running on Apple Silicon using MLX.
Goal: lowest val_bpb in 5 minutes.

## Rules

1. ONLY modify `train.py` (from-scratch mode) or `finetune.py` (fine-tuning mode)
2. Do NOT modify `prepare.py`, `export.py`, or `models.py`
3. Experiments must complete within 5 minutes wall-clock
4. Compare val_bpb (train.py) or val_loss (finetune.py) to previous best
5. Append results to `experiments/log.jsonl`

## Choose Your Mode

**Mode A: Train from scratch** (default, like Karpathy's original)
```
uv run train.py
```
Best when: exploring novel architectures, MoE designs, optimizer research

**Mode B: Fine-tune existing model**
```
python models.py                    # see what models fit your machine
python models.py --prep-data alpaca # prepare a dataset
uv run finetune.py                  # fine-tune with LoRA
```
Best when: customizing a model for a task, domain adaptation, instruction tuning

## Loop

```
# Train from scratch
uv run train.py
cat experiments/log.jsonl | tail -5

# OR fine-tune
uv run finetune.py
cat experiments/log.jsonl | tail -5
```

## Strategic Context

This machine has massive unified memory but fewer FLOPS than NVIDIA.
**Do NOT** try to match NVIDIA speed on the same experiment.
**DO** run experiments a 24GB NVIDIA card cannot fit.

### Priority 1: MoE Architecture Search
MoE is ideal for unified memory -- all experts loaded, only subset activated per token:
- Replace MLP with MoE MLP (top-k gated routing)
- Sweep: 4/8/16/32 experts, top-k=1 or 2, expert sizes
- Try shared expert + routed expert hybrids (DeepSeek style)
- Expert load balancing loss terms

### Priority 2: Large Batch Scaling Laws
Unified memory enables true large batches (not gradient accumulation):
- NVIDIA at 24GB caps out at batch_size~32 for 125M model at seq_len=1024
- We can do batch_size=128, 256, 512
- Research: critical batch size, large batch convergence dynamics

### Priority 3: Full Precision Ablations
Others quantize to 4-bit to fit. We train at full bfloat16:
- Compare bfloat16 vs float32 training
- Larger model at bfloat16 vs smaller model that fits NVIDIA
- Does precision matter more than size when memory isn't the bottleneck?

### Priority 4: Long Context
KV caches eat VRAM. With 128GB+ we can push context:
- max_seq_len: 2048, 4096, 8192
- RoPE theta scaling for extended context

### Priority 5: Width vs Depth
Apple Silicon is memory-bandwidth-bound:
- Wider models (large dim, few layers) may be better than deep narrow ones
- At fixed param count, sweep dim vs n_layers ratio
- GQA head count optimization

### Fine-Tuning Research Priorities (finetune.py mode)

### FT Priority 1: LoRA Rank Scaling on Large Models
- Explore LoRA rank scaling: does rank=64 on a 70B model beat rank=8?
- Compare: QLoRA on 70B vs full LoRA on 7B
- Try DoRA vs standard LoRA vs full fine-tuning

### FT Priority 2: Full Fine-Tuning of Small Models
- On 64GB+ machines, full fine-tuning of 3B-7B models is feasible
- Compare: full fine-tune of 3B vs QLoRA of 14B

### FT Priority 3: Long Context Fine-Tuning
- Increase max_seq_length to 4096, 8192
- KV caches for long sequences eat VRAM -- unified memory handles it

## MLX Idioms

```python
# Compile the full training step
@mx.compile
def train_step(x, y): ...

# Force eval for timing
mx.eval(loss)

# Use fused ops
mx.fast.rms_norm(x, weight, eps)
mx.fast.scaled_dot_product_attention(q, k, v, scale=s, mask=m)
```

## Decision Framework

Before each experiment ask:
1. Could a 24GB NVIDIA card run this? If yes -> probably not our advantage
2. Does this exploit memory abundance? If yes -> high-value experiment
3. Am I compute-bound or memory-bound? Favor memory-bound experiments.
