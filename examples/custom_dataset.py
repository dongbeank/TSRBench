"""
Custom dataset walkthrough for TSRBench.

This example shows how to:
1. Prepare your own CSV for noise injection
2. Customize noise parameters (frequency, duration, amplitude)
3. Use different SPOT algorithm variants
4. Inject noise at a single severity level via the Python API
"""
import numpy as np
import pandas as pd
from tsrbench import CollectiveNoise


def example_single_column():
    """Inject noise into a single column (low-level API)."""
    # Create synthetic data
    np.random.seed(42)
    t = np.arange(1000)
    signal = np.sin(2 * np.pi * t / 200) + 0.1 * np.random.randn(1000)

    cn = CollectiveNoise(seed=2025)

    # Inject level shift at severity level 3
    shift_noise = cn.inject_level_shift(signal, noise_level=3)
    shifted_signal = signal + shift_noise

    # Inject exponential spike at severity level 3
    spike_noise = cn.inject_exp_spike(signal, noise_level=3)
    spiked_signal = signal + spike_noise

    print(f"Original range:  [{signal.min():.3f}, {signal.max():.3f}]")
    print(f"Shifted range:   [{shifted_signal.min():.3f}, {shifted_signal.max():.3f}]")
    print(f"Spiked range:    [{spiked_signal.min():.3f}, {spiked_signal.max():.3f}]")


def example_custom_parameters():
    """Use custom noise parameters instead of defaults."""
    custom_shift = {
        1: {'freq': 0.001, 'dur': 4, 'amp': 0.002},
        2: {'freq': 0.003, 'dur': 8, 'amp': 0.001},
        3: {'freq': 0.005, 'dur': 12, 'amp': 0.0005},
    }
    custom_spike = {
        1: {'freq': 0.001, 'dur': 4, 'amp': 0.002},
        2: {'freq': 0.003, 'dur': 8, 'amp': 0.001},
        3: {'freq': 0.005, 'dur': 12, 'amp': 0.0005},
    }
    custom_spot = {
        'type': 'bispot',      # Use biSPOT instead of bidSPOT
        'n_points': 10,
        'depth': 0.01,
        'init_points': 0.05,
        'init_level': 0.98,
    }

    cn = CollectiveNoise(
        seed=2025,
        level_shift_args=custom_shift,
        exp_spike_args=custom_spike,
        spot_args=custom_spot,
    )

    np.random.seed(42)
    signal = np.sin(np.linspace(0, 20, 2000)) + 0.05 * np.random.randn(2000)

    shift, spike = cn.inject_noise(signal, noise_level=2)
    print(f"Custom params - shift noise points: {np.count_nonzero(shift)}, spike noise points: {np.count_nonzero(spike)}")


def example_full_dataset():
    """
    Generate corrupted versions of a full CSV dataset.

    Your CSV should have:
    - First column: timestamps (date, index, etc.)
    - Remaining columns: numeric time series values

    Example CSV format:
        date,feature1,feature2,feature3
        2020-01-01,1.23,4.56,7.89
        2020-01-02,1.45,4.67,8.01
        ...
    """
    import os
    import tempfile

    # Create a sample dataset
    np.random.seed(42)
    n = 5000
    dates = pd.date_range('2020-01-01', periods=n, freq='h')
    df = pd.DataFrame({
        'date': dates,
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.randn(n),
        'humidity': 60 + 15 * np.sin(2 * np.pi * np.arange(n) / 168) + 2 * np.random.randn(n),
        'pressure': 1013 + 5 * np.sin(2 * np.pi * np.arange(n) / 720) + 0.5 * np.random.randn(n),
    })

    # Save to a temp directory
    tmpdir = tempfile.mkdtemp()
    csv_path = os.path.join(tmpdir, 'my_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Sample dataset saved to: {csv_path}")

    # Create args object for make_noise_datasets
    class Args:
        data_path = 'my_data.csv'
        root_path = tmpdir
        output_path = os.path.join(tmpdir, 'noisy')
        spot_type = 'bidspot'
        spot_n_points = 8
        spot_depth = 0.01
        spot_init_points = 0.05
        spot_init_level = 0.98
        zero_clip = False

    cn = CollectiveNoise(seed=2025)
    cn.make_noise_datasets(Args())

    # List generated files
    noisy_dir = os.path.join(tmpdir, 'noisy')
    print(f"\nGenerated files in {noisy_dir}:")
    for f in sorted(os.listdir(noisy_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Single column noise injection")
    print("=" * 60)
    example_single_column()

    print()
    print("=" * 60)
    print("Example 2: Custom noise parameters")
    print("=" * 60)
    example_custom_parameters()

    print()
    print("=" * 60)
    print("Example 3: Full dataset generation")
    print("=" * 60)
    example_full_dataset()
