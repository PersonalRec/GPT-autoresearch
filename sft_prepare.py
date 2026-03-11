"""
One-time preparation script for SFT training.
Downloads and validates the model + dataset, and runs a quick sanity check.

Usage:
    uv run sft_prepare.py
    uv run sft_prepare.py --check-only   # only verify what's already downloaded
"""

import argparse
import os
import sys
import time

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "vngrs-ai/Kumru-2B-Base"
DATASET_NAME = "oztrkoguz/Open_Math_Instruct_Turkish"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_sft")
os.makedirs(CACHE_DIR, exist_ok=True)


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found. SFT training requires a GPU.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    if vram_gb < 6:
        print("WARNING: Less than 6 GB VRAM detected.")
        print("  Recommendation: set USE_LORA=True and lower MAX_SEQ_LEN in sft_train.py")
    elif vram_gb < 12:
        print("  Recommendation: set USE_LORA=True or lower DEVICE_BATCH_SIZE in sft_train.py")
    return vram_gb


def download_tokenizer():
    from transformers import AutoTokenizer
    print(f"\nDownloading tokenizer: {MODEL_NAME}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=os.path.join(CACHE_DIR, "model"),
    )
    print(f"  Tokenizer downloaded in {time.time() - t0:.1f}s")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    if tokenizer.chat_template:
        print("  Chat template: available")
    else:
        print("  Chat template: not found (will use fallback format)")

    # Quick encode/decode test
    test_text = "Merhaba dünya! 2 + 2 = 4"
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"  Tokenizer sanity check: '{test_text}' -> {len(ids)} tokens -> '{decoded}'")
    return tokenizer


def download_dataset():
    from datasets import load_dataset
    print(f"\nDownloading dataset: {DATASET_NAME}")
    t0 = time.time()
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=os.path.join(CACHE_DIR, "data"),
    )
    print(f"  Dataset downloaded in {time.time() - t0:.1f}s")
    print(f"  Total examples: {len(dataset):,}")
    print(f"  Columns: {dataset.column_names}")

    # Show sample
    print("\n  Sample example:")
    example = dataset[0]
    q = example["question"][:200] + ("..." if len(example["question"]) > 200 else "")
    a = example["answer"][:200] + ("..." if len(example["answer"]) > 200 else "")
    print(f"  Question: {q}")
    print(f"  Answer:   {a}")

    # Distribution stats
    q_lens = [len(ex["question"]) for ex in dataset]
    a_lens = [len(ex["answer"]) for ex in dataset]
    print(f"\n  Question length: min={min(q_lens)}, max={max(q_lens)}, mean={sum(q_lens)//len(q_lens)}")
    print(f"  Answer length:   min={min(a_lens)}, max={max(a_lens)}, mean={sum(a_lens)//len(a_lens)}")
    return dataset


def check_tokenization_stats(tokenizer, dataset, seq_len=512):
    """Compute token length distribution to help choose MAX_SEQ_LEN."""
    print(f"\nComputing token length statistics (sample of 200 examples)...")
    from sft_train import format_conversation

    lengths = []
    for i, example in enumerate(dataset):
        if i >= 200:
            break
        text = format_conversation(example, tokenizer)
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))

    lengths.sort()
    p50 = lengths[len(lengths) // 2]
    p90 = lengths[int(len(lengths) * 0.90)]
    p95 = lengths[int(len(lengths) * 0.95)]
    p99 = lengths[int(len(lengths) * 0.99)]
    print(f"  Token lengths — p50: {p50}, p90: {p90}, p95: {p95}, p99: {p99}, max: {max(lengths)}")
    print(f"  Current MAX_SEQ_LEN={seq_len}")
    if seq_len < p90:
        print(f"  WARNING: {int(100*(1 - sum(1 for l in lengths if l <= seq_len)/len(lengths)))}% of examples will be truncated at seq_len={seq_len}")
        print(f"  Recommendation: set MAX_SEQ_LEN >= {p90} in sft_train.py")
    else:
        pct_fits = 100 * sum(1 for l in lengths if l <= seq_len) / len(lengths)
        print(f"  {pct_fits:.1f}% of examples fit within MAX_SEQ_LEN={seq_len}")


def estimate_training_steps(dataset_size, batch_size, time_budget, steps_per_sec_estimate=0.3):
    """Rough estimate of how many training steps will run."""
    steps = int(time_budget * steps_per_sec_estimate)
    examples_per_step = batch_size
    total_examples = steps * examples_per_step
    epochs = total_examples / dataset_size
    print(f"\nTraining estimate (rough):")
    print(f"  ~{steps} optimizer steps in {time_budget}s")
    print(f"  ~{epochs:.2f} epochs over {dataset_size} examples")
    print(f"  (Actual speed depends on GPU, seq_len, batch_size)")


def main():
    parser = argparse.ArgumentParser(description="Prepare for SFT training")
    parser.add_argument("--check-only", action="store_true",
                        help="Only verify existing downloads, don't download")
    args = parser.parse_args()

    print("=" * 60)
    print("SFT Preparation: Kumru-2B-Base + Open_Math_Instruct_Turkish")
    print("=" * 60)
    print(f"Cache directory: {CACHE_DIR}")

    # 1. GPU check
    vram_gb = check_gpu()

    # 2. Tokenizer
    tokenizer = download_tokenizer()

    # 3. Dataset
    dataset = download_dataset()

    # 4. Tokenization stats
    try:
        from sft_train import MAX_SEQ_LEN
        check_tokenization_stats(tokenizer, dataset, seq_len=MAX_SEQ_LEN)
    except ImportError:
        print("\nCould not import sft_train.py — skipping tokenization stats")

    # 5. Training estimate
    try:
        from sft_train import TOTAL_BATCH_SIZE, TIME_BUDGET
        estimate_training_steps(len(dataset), TOTAL_BATCH_SIZE, TIME_BUDGET)
    except ImportError:
        pass

    # 6. Recommendations
    print("\n" + "=" * 60)
    print("Configuration recommendations:")
    print("=" * 60)
    if vram_gb < 12:
        print("  ✓ Set USE_LORA=True  (saves VRAM by training only adapter weights)")
        print("  ✓ Lower DEVICE_BATCH_SIZE to 2 or 1 if you get OOM errors")
    if vram_gb < 8:
        print("  ✓ Lower MAX_SEQ_LEN to 256")
    print("  ✓ Run: uv run sft_train.py")
    print()
    print("Done! Ready for SFT training.")


if __name__ == "__main__":
    main()
