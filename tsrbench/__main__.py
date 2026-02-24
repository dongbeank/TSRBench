"""
CLI entry point for TSRBench.

Usage:
    python -m tsrbench --data-path my_data.csv --root-path ./data/ --output-path ./data/noisy/
"""
from tsrbench.collective_noise import CollectiveNoise
import argparse


def main():
    parser = argparse.ArgumentParser(description='TSRBench: Time Series Robust Benchmark - Noise Injection')
    parser.add_argument('--data-path', type=str, required=True, help='dataset filename (e.g. ETTh1.csv)')
    parser.add_argument('--root-path', type=str, required=True, help='root path containing the original CSV')
    parser.add_argument('--output-path', type=str, default=None, help='output path for noise files (default: same as root-path)')
    parser.add_argument('--spot-type', type=str, default='bidspot',
                        choices=['spot', 'bispot', 'dspot', 'bidspot'],
                        help='SPOT algorithm variant (default: bidspot)')
    parser.add_argument('--spot-n-points', type=int, default=8, help='number of extreme points for SPOT (default: 8)')
    parser.add_argument('--spot-depth', type=float, default=0.01, help='depth for dSPOT/bidSPOT (default: 0.01)')
    parser.add_argument('--spot-init-points', type=float, default=0.05, help='fraction of data for SPOT initialization (default: 0.05)')
    parser.add_argument('--spot-init-level', type=float, default=0.98, help='initial level for SPOT (default: 0.98)')
    parser.add_argument('--zero-clip', action='store_true', default=False, help='clip negative values to zero')
    parser.add_argument('--seed', type=int, default=2025, help='random seed (default: 2025)')
    args = parser.parse_args()

    cn = CollectiveNoise(seed=args.seed)
    print(f"TSRBench: Noise Injection into {args.data_path}")
    cn.make_noise_datasets(args)


if __name__ == "__main__":
    main()
