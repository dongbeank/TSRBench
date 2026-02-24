"""
Visualization utilities for TSRBench corrupted time series.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_pair(original_csv, noise_dir, data_name, level, noise_type):
    """Load original and corrupted DataFrames."""
    original_df = pd.read_csv(original_csv)
    noise_file = os.path.join(noise_dir, f"{data_name}_level_{level}_type_{noise_type}.csv")
    noise_df = pd.read_csv(noise_file)
    return original_df, noise_df


def _infer_data_name(original_csv):
    """Infer dataset name from CSV filename (e.g. 'ETTh1.csv' -> 'ETTh1')."""
    return os.path.splitext(os.path.basename(original_csv))[0]


def plot_corruption_comparison(original_csv, noise_dir, column, level=3,
                                save_path=None, figsize=(24, 4)):
    """
    Side-by-side 4-panel plot: Original | Level Shift | Spike | Combined.

    Parameters
    ----------
    original_csv : str
        Path to the original CSV file.
    noise_dir : str
        Directory containing the corrupted CSV files.
    column : str
        Column name to visualize.
    level : int
        Severity level (1-5).
    save_path : str or None
        If provided, save figure to this path.
    figsize : tuple
        Figure size (width, height) per panel row.
    """
    data_name = _infer_data_name(original_csv)
    original_df = pd.read_csv(original_csv)

    noise_types = ['shift', 'spike', 'combined']
    titles = ['Original', 'Level Shift', 'Exponential Spike', 'Combined']

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)

    # Original
    axes[0].plot(original_df[column].values, color='#2196F3', linewidth=0.5)
    axes[0].set_title(titles[0], fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel(column)

    # Corrupted versions
    for i, ntype in enumerate(noise_types):
        noise_file = os.path.join(noise_dir, f"{data_name}_level_{level}_type_{ntype}.csv")
        if not os.path.exists(noise_file):
            axes[i + 1].text(0.5, 0.5, f'File not found:\n{os.path.basename(noise_file)}',
                           ha='center', va='center', transform=axes[i + 1].transAxes)
            axes[i + 1].set_title(titles[i + 1], fontsize=12, fontweight='bold')
            continue
        noise_df = pd.read_csv(noise_file)
        axes[i + 1].plot(noise_df[column].values, color='#F44336', linewidth=0.5, alpha=0.8)
        axes[i + 1].plot(original_df[column].values, color='#2196F3', linewidth=0.3, alpha=0.4)
        axes[i + 1].set_title(titles[i + 1], fontsize=12, fontweight='bold')
        axes[i + 1].set_xlabel('Time Step')

    fig.suptitle(f'{data_name} - {column} - Level {level}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_severity_levels(original_csv, noise_dir, column, noise_type='combined',
                          save_path=None, figsize=(24, 12)):
    """
    5-panel plot showing severity levels 1-5 for a given noise type.

    Parameters
    ----------
    original_csv : str
        Path to the original CSV file.
    noise_dir : str
        Directory containing the corrupted CSV files.
    column : str
        Column name to visualize.
    noise_type : str
        Noise type: 'shift', 'spike', or 'combined'.
    save_path : str or None
        If provided, save figure to this path.
    figsize : tuple
        Figure size.
    """
    data_name = _infer_data_name(original_csv)
    original_df = pd.read_csv(original_csv)

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True, sharey=True)

    for level in range(1, 6):
        ax = axes[level - 1]
        noise_file = os.path.join(noise_dir, f"{data_name}_level_{level}_type_{noise_type}.csv")

        ax.plot(original_df[column].values, color='#2196F3', linewidth=0.5, alpha=0.5, label='Original')

        if os.path.exists(noise_file):
            noise_df = pd.read_csv(noise_file)
            ax.plot(noise_df[column].values, color='#F44336', linewidth=0.5, alpha=0.8, label=f'Level {level}')
        else:
            ax.text(0.5, 0.5, f'File not found', ha='center', va='center', transform=ax.transAxes)

        ax.set_ylabel(column, fontsize=9)
        ax.set_title(f'Severity Level {level}', fontsize=11, fontweight='bold', loc='left')
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time Step')
    type_labels = {'shift': 'Level Shift', 'spike': 'Exponential Spike', 'combined': 'Combined'}
    fig.suptitle(f'{data_name} - {column} - {type_labels.get(noise_type, noise_type)}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_noise_only(original_csv, noise_dir, column, level=3,
                     save_path=None, figsize=(20, 6)):
    """
    Show the isolated noise signal (corrupted - original) for shift and spike.

    Parameters
    ----------
    original_csv : str
        Path to the original CSV file.
    noise_dir : str
        Directory containing the corrupted CSV files.
    column : str
        Column name to visualize.
    level : int
        Severity level (1-5).
    save_path : str or None
        If provided, save figure to this path.
    figsize : tuple
        Figure size.
    """
    data_name = _infer_data_name(original_csv)
    original_df = pd.read_csv(original_csv)
    original_vals = original_df[column].values

    noise_types = ['shift', 'spike']
    titles = ['Level Shift Noise', 'Exponential Spike Noise']
    colors = ['#FF9800', '#9C27B0']

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

    for i, ntype in enumerate(noise_types):
        ax = axes[i]
        noise_file = os.path.join(noise_dir, f"{data_name}_level_{level}_type_{ntype}.csv")

        if os.path.exists(noise_file):
            noise_df = pd.read_csv(noise_file)
            noise_signal = noise_df[column].values - original_vals
            ax.fill_between(range(len(noise_signal)), noise_signal, 0, alpha=0.6, color=colors[i])
            ax.plot(noise_signal, color=colors[i], linewidth=0.3)
        else:
            ax.text(0.5, 0.5, 'File not found', ha='center', va='center', transform=ax.transAxes)

        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Noise Amplitude')
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    fig.suptitle(f'{data_name} - {column} - Isolated Noise (Level {level})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig
