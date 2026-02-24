"""
Visualization examples for TSRBench.

This script demonstrates the three visualization functions:
1. plot_corruption_comparison - side-by-side view of all corruption types
2. plot_severity_levels - progression across severity levels 1-5
3. plot_noise_only - isolated noise signal visualization

Usage:
    python examples/visualize_corruptions.py \
        --original dataset/ETT-small/ETTh1.csv \
        --noise-dir dataset/ETT-small/ETTh1_noise/ \
        --column HUFL
"""
import argparse
from tsrbench import plot_corruption_comparison, plot_severity_levels, plot_noise_only


def main():
    parser = argparse.ArgumentParser(description='TSRBench visualization examples')
    parser.add_argument('--original', type=str, required=True, help='path to original CSV')
    parser.add_argument('--noise-dir', type=str, required=True, help='directory with corrupted CSVs')
    parser.add_argument('--column', type=str, required=True, help='column name to visualize')
    parser.add_argument('--level', type=int, default=3, help='severity level for comparison/noise plots (default: 3)')
    parser.add_argument('--save-dir', type=str, default=None, help='directory to save figures (default: show only)')
    args = parser.parse_args()

    save_prefix = None
    if args.save_dir:
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        save_prefix = args.save_dir

    # 1. Corruption comparison (Original | Shift | Spike | Combined)
    print("Generating corruption comparison plot...")
    plot_corruption_comparison(
        args.original, args.noise_dir, args.column,
        level=args.level,
        save_path=f"{save_prefix}/comparison_{args.column}_L{args.level}.png" if save_prefix else None,
    )

    # 2. Severity levels (all 5 levels for one noise type)
    print("Generating severity levels plot...")
    for noise_type in ['shift', 'spike', 'combined']:
        plot_severity_levels(
            args.original, args.noise_dir, args.column,
            noise_type=noise_type,
            save_path=f"{save_prefix}/severity_{args.column}_{noise_type}.png" if save_prefix else None,
        )

    # 3. Isolated noise signal
    print("Generating isolated noise plot...")
    plot_noise_only(
        args.original, args.noise_dir, args.column,
        level=args.level,
        save_path=f"{save_prefix}/noise_only_{args.column}_L{args.level}.png" if save_prefix else None,
    )

    print("Done!")
    if not save_prefix:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
