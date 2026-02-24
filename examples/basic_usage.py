"""
Basic usage of TSRBench: inject noise into time series data.

This example demonstrates two approaches:
1. Python API (corrupt method) — for programmatic use
2. CSV batch processing (make_noise_datasets) — for generating all levels at once
"""
from tsrbench import CollectiveNoise
import numpy as np
import pandas as pd
import argparse


def python_api_example():
    """Demonstrate the Python API with corrupt()."""
    print("=== Python API Example ===\n")

    # 1D signal
    signal = np.random.randn(5000)
    cn = CollectiveNoise(seed=2025)
    results = cn.corrupt(signal, noise_level=3)
    print(f"1D input shape: {signal.shape}")
    print(f"Output keys: {list(results.keys())}")
    print(f"Shift output shape: {results['shift'].shape}\n")

    # DataFrame with date column
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5000, freq='h'),
        'temperature': np.random.randn(5000) * 10 + 20,
        'humidity': np.random.randn(5000) * 5 + 60,
    })
    cn = CollectiveNoise(seed=2025)
    results = cn.corrupt(df, noise_level=3, skip_first_col=True)
    print(f"DataFrame input shape: {df.shape}")
    print(f"Shift DataFrame shape: {results['shift'].shape}")
    print(f"Date column preserved: {(results['shift']['date'] == df['date']).all()}\n")

    # 2D numpy array
    X = np.random.randn(3000, 5)
    cn = CollectiveNoise(seed=2025)
    results = cn.corrupt(X, noise_level=3, skip_first_col=False)
    print(f"2D array input shape: {X.shape}")
    print(f"Combined output shape: {results['combined'].shape}\n")


def csv_batch_example():
    """Demonstrate batch CSV generation."""
    parser = argparse.ArgumentParser(description='TSRBench basic usage example')
    parser.add_argument('--root-path', type=str, required=True, help='directory containing the CSV')
    parser.add_argument('--data-path', type=str, required=True, help='CSV filename (e.g. ETTh1.csv)')
    parser.add_argument('--output-path', type=str, default=None, help='output directory (default: same as root-path)')
    args = parser.parse_args()

    # Fill in defaults expected by make_noise_datasets
    args.spot_type = 'bidspot'
    args.spot_n_points = 8
    args.spot_depth = 0.01
    args.spot_init_points = 0.05
    args.spot_init_level = 0.98
    args.zero_clip = False

    cn = CollectiveNoise(seed=2025)
    cn.make_noise_datasets(args)
    # Output: <data>_level_{1..5}_type_{shift,spike,impulse,gaussian,missing,combined}.csv


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and '--root-path' in sys.argv:
        csv_batch_example()
    else:
        python_api_example()
