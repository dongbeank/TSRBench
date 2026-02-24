#!/bin/bash

# Generate corrupted benchmark datasets for TSRBench
# Reproduces the noise injection used in the paper for all 6 standard datasets.
#
# Prerequisites:
#   1. pip install -e .   (install tsrbench)
#   2. Download datasets and place them under ./dataset/
#      - ETT-small/: ETTm1.csv, ETTm2.csv, ETTh1.csv, ETTh2.csv
#      - electricity/: electricity.csv
#      - weather/: weather.csv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/dataset}"

echo "=== TSRBench: Generating corrupted benchmark datasets ==="
echo "Dataset directory: $DATASET_DIR"

# ETT datasets (spot_depth=0.01)
for name in ETTm1 ETTm2 ETTh1 ETTh2; do
    echo ""
    echo "--- Noise injection: ${name} ---"
    python -m tsrbench \
        --data-path "${name}.csv" \
        --root-path "$DATASET_DIR/ETT-small/" \
        --output-path "$DATASET_DIR/ETT-small/${name}_noise/" \
        --spot-type bidspot \
        --spot-n-points 8 \
        --spot-depth 0.01
done

# Electricity (spot_depth=0.02)
echo ""
echo "--- Noise injection: electricity ---"
python -m tsrbench \
    --data-path electricity.csv \
    --root-path "$DATASET_DIR/electricity/" \
    --output-path "$DATASET_DIR/electricity/electricity_noise/" \
    --spot-type bidspot \
    --spot-n-points 8 \
    --spot-depth 0.02

# Weather (spot_depth=0.01)
echo ""
echo "--- Noise injection: weather ---"
python -m tsrbench \
    --data-path weather.csv \
    --root-path "$DATASET_DIR/weather/" \
    --output-path "$DATASET_DIR/weather/weather_noise/" \
    --spot-type bidspot \
    --spot-n-points 8 \
    --spot-depth 0.01

echo ""
echo "=== Done! All benchmark datasets generated. ==="
