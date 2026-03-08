# autoresearch_mlx

Autonomous AI research on Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx).

A native MLX rewrite of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Give an AI agent a real LLM training setup on your Mac and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

## Why MLX, not PyTorch MPS?

PyTorch's MPS backend is a compatibility shim -- it maps CUDA abstractions onto Metal. MLX is built from the ground up for Apple Silicon. The difference matters:

| Feature | PyTorch MPS | MLX |
|---------|-------------|-----|
| torch.compile | Disabled | mx.compile works |
| FlashAttention | Unavailable | mx.fast.scaled_dot_product_attention |
| Memory model | CUDA-style (.to()) | Unified (zero-copy) |
| Precision | fp16 ~20% speedup | bfloat16 native |

With MLX, Apple Silicon becomes a genuine research platform rather than a second-class citizen running someone else's abstractions.

## The Memory Advantage Thesis

Apple Silicon's unified memory architecture shares a single pool between CPU and GPU -- no PCIe copies, no VRAM limits separate from system RAM. A MacBook Pro with 128GB unified memory can fit models and batch sizes that a 24GB NVIDIA card simply cannot. This project leans into that advantage:

- **MoE architectures**: All experts resident in memory, only a subset activated per token
- **Large batch training**: batch_size=256 or 512 where NVIDIA caps out at 32
- **Full precision**: Train at bfloat16 without quantizing to 4-bit just to fit
- **Long context**: KV caches for 8192-token sequences without OOM

The tradeoff: fewer FLOPS than NVIDIA. So we run experiments that exploit memory abundance, not raw compute speed.

## Choose Your Adventure

### Mode A: Train from scratch (default)

Train a GPT model from scratch on the ClimbMix dataset, just like Karpathy's original:

```bash
# 1. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 2. Train for 5 minutes
uv run train.py
```

The agent modifies `train.py` -- architecture, optimizer, hyperparameters, batch size, everything is fair game.

### Mode B: Fine-tune an existing model

Fine-tune a pretrained model (Llama, Qwen, Gemma, Mistral, etc.) with LoRA:

```bash
# 1. See what models fit your machine
python models.py

# 2. Prepare a dataset
python models.py --prep-data alpaca

# 3. Fine-tune with LoRA
uv run finetune.py
```

The agent modifies `finetune.py` -- LoRA config, learning rate, dataset mixing, training schedule.

## Model Selection

Run `python models.py` to see which pretrained models fit your machine's available memory. It lists model families (Llama, Qwen, Gemma, Mistral, etc.) with their parameter counts and memory requirements, and recommends the largest model your hardware can handle.

## Dataset Handling

For from-scratch training, `prepare.py` downloads and tokenizes the ClimbMix dataset automatically.

For fine-tuning, `models.py --prep-data <preset>` supports built-in presets (alpaca, dolly, oasst, etc.) as well as any HuggingFace dataset. Data is cached in `~/.cache/autoresearch/`.

## Export to Ollama

After training or fine-tuning, export your model for local inference with Ollama:

```bash
python export.py experiments/<experiment-id>/
```

This packages the weights and tokenizer into a format Ollama can serve directly.

## Project Structure

```
train.py        -- from-scratch model + training loop (agent modifies this)
finetune.py     -- LoRA fine-tuning script (agent modifies this)
prepare.py      -- data prep + runtime utilities (do not modify)
models.py       -- model catalog + data prep for fine-tuning (do not modify)
export.py       -- export to Ollama format (do not modify)
program.md      -- agent instructions (human modifies this)
pyproject.toml  -- dependencies
```

## Running the Agent

Spin up Claude Code (or any agent) in this repo, then prompt:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent reads `program.md`, picks an experiment strategy, and loops autonomously -- modifying code, training, evaluating, keeping or discarding, and repeating. Each experiment takes ~5 minutes, so you get ~12 experiments per hour, or ~100 overnight.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4, any tier)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Design Choices

- **Single file to modify.** The agent only touches `train.py` (or `finetune.py`). This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes. This makes experiments directly comparable regardless of what the agent changes. autoresearch finds the most optimal model for your specific hardware in that budget.
- **Native MLX.** No PyTorch, no compatibility layers. Direct Apple Silicon Metal compute via MLX's C++ runtime.

## Credits

- [Karpathy](https://github.com/karpathy/autoresearch) -- the original autoresearch concept and training setup
- [miolini](https://github.com/miolini/autoresearch-macos) -- macOS fork that proved Apple Silicon viability
- [Apple MLX team](https://github.com/ml-explore/mlx) -- the MLX framework

## License

MIT
