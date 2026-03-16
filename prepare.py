"""
One-time data preparation for autoresearch experiments.
Streams FineWeb-Edu data and sets up GPT-2 tokenizer.

Usage:
    python prepare.py                    # full prep (100K docs + tokenizer)
    python prepare.py --num-docs 10000   # fewer docs (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import argparse

# Fixing of the HuggingFace issue with slow dataset downloading 
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Disable legacy fast transfer
os.environ["HF_HUB_DISABLE_XET"] = "1"         # Disable Xet backend (v1.0+)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes (default is 10s)
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"       # Metadata timeout

import time
import math
import pickle
import pyarrow.parquet as pq
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024       # context length
TIME_BUDGET = 1200        # training time budget in seconds
EVAL_TOKENS = 5 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# FineWeb-Edu dataset
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
NUM_DOCS = 200_000              # ~100M tokens (1% of 10BT sample)
DOCS_PER_SHARD = 100_000        # documents per parquet shard

VAL_FILENAME = "val.parquet"
EOT_TOKEN = "<|endoftext|>"     # GPT-2 delimiter / BOS token

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(num_docs=NUM_DOCS):
    """Stream FineWeb-Edu, take num_docs documents, split into train/val, save as parquet shards."""
    os.makedirs(DATA_DIR, exist_ok=True)

    existing = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    if len(existing) >= 2:
        print(f"Data: {len(existing)} shards already at {DATA_DIR}")
        return

    import pyarrow as pa
    from datasets import load_dataset

    print(f"Streaming {DATASET_NAME} ({DATASET_CONFIG}), taking {num_docs:,} docs...")
    t0 = time.time()
    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)
    docs = list(ds.take(num_docs))
    del ds  # close streaming connection
    t1 = time.time()
    print(f"  Downloaded {len(docs):,} documents in {t1 - t0:.1f}s")

    # Split: 99% train, 1% val
    import random
    random.seed(42)
    random.shuffle(docs)
    val_count = max(1, len(docs) // 100)
    val_docs = docs[:val_count]
    train_docs = docs[val_count:]

    # Save val shard
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    val_table = pa.table({"text": [d["text"] for d in val_docs]})
    pq.write_table(val_table, val_path)
    print(f"  Val: {len(val_docs):,} docs -> {val_path}")

    # Save train shards
    num_shards = max(1, (len(train_docs) + DOCS_PER_SHARD - 1) // DOCS_PER_SHARD)
    for i in range(num_shards):
        start = i * DOCS_PER_SHARD
        end = min(start + DOCS_PER_SHARD, len(train_docs))
        shard_docs = train_docs[start:end]
        shard_table = pa.table({"text": [d["text"] for d in shard_docs]})
        shard_path = os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")
        pq.write_table(shard_table, shard_path)
        print(f"  Train shard {i}: {end - start:,} docs -> {shard_path}")

    print(f"Data: {len(train_docs):,} train docs in {num_shards} shards, ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer setup (GPT-2, no training needed)
# ---------------------------------------------------------------------------

def setup_tokenizer():
    """Save GPT-2 tokenizer and token_bytes lookup for BPB evaluation."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already set up at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    # Save tokenizer as pickle (same format as before)
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Tokenizer: GPT-2 saved to {tokenizer_pkl} (vocab_size={enc.n_vocab})")

    # Build token_bytes lookup for BPB evaluation
    print("Tokenizer: building token_bytes lookup...")
    eot_id = enc._special_tokens[EOT_TOKEN]
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        if token_id == eot_id:
            token_bytes_list.append(0)  # special tokens excluded from BPB
        else:
            token_bytes_list.append(len(enc.decode([token_id]).encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print("Tokenizer: sanity check passed")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper around tiktoken GPT-2 encoding."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc._special_tokens[EOT_TOKEN]

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-docs", type=int, default=NUM_DOCS,
                        help=f"Number of documents to download (default: {NUM_DOCS:,})")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Stream FineWeb-Edu data
    download_data(num_docs=args.num_docs)
    print()

    # Step 2: Set up GPT-2 tokenizer
    setup_tokenizer()
    print()
    print("Done! Ready to train.")
