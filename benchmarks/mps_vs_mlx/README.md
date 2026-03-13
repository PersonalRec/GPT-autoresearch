# Benchmark: PyTorch MPS vs MLX

Comparison of Apple Silicon training approaches:
- **PyTorch MPS** — unified `train.py` with GPU Muon
- **MLX** — PR #202 `train_mlx.py` with CPU Muon

## Results

| Metric | PyTorch MPS | MLX | Δ |
|--------|-------------|-----|---|
| Throughput | 12,531 tok/s | 6,946 tok/s | **1.80x** |
| Optimizer | 51 ms | 202 ms | **3.98x** |

## Structure

```
benchmarks/mps_vs_mlx/
├── README.md
├── src/
│   ├── run.py              # Main benchmark runner
│   ├── plot.py             # Plot generator
│   ├── config.py           # Configuration
│   └── runners/
│       ├── pytorch_mps.py  # PyTorch MPS benchmark
│       └── mlx_cpu_muon.py # MLX benchmark
├── results/
│   └── benchmark.json      # Raw data
├── plots/
│   ├── throughput.png
│   ├── breakdown.png
│   ├── optimizer.png
│   └── summary.png
└── reference/
    └── train_mlx_pr202.py  # PR #202 original code
```

## Usage

```bash
# Run benchmark (requires mlx)
uv sync --extra benchmark
uv run benchmarks/mps_vs_mlx/src/run.py

# Generate plots
uv run benchmarks/mps_vs_mlx/src/plot.py

# Custom config
uv run benchmarks/mps_vs_mlx/src/run.py --steps 100 --batch-size 8
```

## Conclusion

PyTorch MPS is 1.8x faster due to GPU Muon (4x faster than CPU Muon).
