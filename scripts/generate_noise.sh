#!/bin/bash

# Generic noise generation script for a single custom dataset.
#
# Usage:
#   bash scripts/generate_noise.sh <root_path> <data_file> [output_path] [spot_depth]
#
# Example:
#   bash scripts/generate_noise.sh ./dataset/my_data/ my_data.csv ./dataset/my_data/noisy/ 0.01
#
# Prerequisites:
#   pip install -e .

ROOT_PATH="${1:?Usage: $0 <root_path> <data_file> [output_path] [spot_depth]}"
DATA_FILE="${2:?Usage: $0 <root_path> <data_file> [output_path] [spot_depth]}"
OUTPUT_PATH="${3:-$ROOT_PATH}"
SPOT_DEPTH="${4:-0.01}"

echo "=== TSRBench: Noise Injection ==="
echo "  Input:  ${ROOT_PATH}/${DATA_FILE}"
echo "  Output: ${OUTPUT_PATH}"
echo "  SPOT depth: ${SPOT_DEPTH}"
echo ""

python -m tsrbench \
    --data-path "$DATA_FILE" \
    --root-path "$ROOT_PATH" \
    --output-path "$OUTPUT_PATH" \
    --spot-type bidspot \
    --spot-n-points 8 \
    --spot-depth "$SPOT_DEPTH"

echo ""
echo "=== Done! ==="
