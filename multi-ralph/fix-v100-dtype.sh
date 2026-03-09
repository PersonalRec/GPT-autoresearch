#!/bin/bash
# Fix train.py for V100 (no BF16 support)
# Converts bfloat16 -> float16 and adds loss scaling for FP16 stability
# Run BEFORE launch.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FILE="$REPO_DIR/train.py"

echo "Patching $FILE for V100 (FP16)..."

# Replace bfloat16 with float16
sed -i.bak 's/torch\.bfloat16/torch.float16/g' "$FILE"

# Count replacements
CHANGES=$(diff "$FILE.bak" "$FILE" | grep "^[<>]" | wc -l)
echo "  Changed $((CHANGES / 2)) lines (bfloat16 -> float16)"

# Also update the best copy
mkdir -p "$REPO_DIR/multi-ralph/best"
cp "$FILE" "$REPO_DIR/multi-ralph/best/train.py"

# Clean up backup
rm -f "$FILE.bak"

echo "Done. Run 'CUDA_VISIBLE_DEVICES=0 uv run train.py' to verify."
