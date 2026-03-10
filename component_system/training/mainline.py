"""Mainline assembler: reads static config, dynamically loads components, runs training."""
from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import importlib
import os
from dataclasses import asdict
from typing import Any

import torch

from prepare import Tokenizer

from component_system.config import get_training_binding


def _prepare_environment() -> None:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()


def _import_module(path: str) -> Any:
    return importlib.import_module(path)


def run_mainline_training(binding_path: str | None = None) -> dict[str, Any]:
    _prepare_environment()
    binding = get_training_binding()

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    model_module = _import_module(binding["model_module"])
    optimizer_module = _import_module(binding["optimizer_module"])
    training_step_module = _import_module(binding["training_step_module"])

    settings = training_step_module.default_training_settings()
    config = model_module.build_model_config(
        depth=settings.depth,
        vocab_size=vocab_size,
        aspect_ratio=settings.aspect_ratio,
        head_dim=settings.head_dim,
        window_pattern=settings.window_pattern,
    )

    print("Loaded training binding from config")
    print(f"Model config: {asdict(config)}")

    model, param_counts, num_flops_per_token = model_module.create_model(
        config,
        compile_model=settings.compile_model,
    )

    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    optimizer = optimizer_module.create_optimizer(model, settings)
    return training_step_module.run_training_session(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        settings=settings,
        param_counts=param_counts,
        num_flops_per_token=num_flops_per_token,
        baseline_binding=binding,
    )


if __name__ == "__main__":
    run_mainline_training()
