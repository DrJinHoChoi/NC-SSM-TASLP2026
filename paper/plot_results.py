#!/usr/bin/env python3
"""
NanoMamba TASLP — Noise Robustness Results Visualization
Generates publication-quality figures from per-sample noise-aug experiment results.

Usage:
    python plot_results.py          # generates 4 PNG figures in paper/ directory
    python plot_results.py --pdf    # generates PDF versions (for LaTeX)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import sys
import os

# ============================================================================
# Experiment Results (Per-Sample Noise-Aug Trained, GSC V2 12-class)
# Order: [-15dB, -10dB, -5dB, 0dB, 5dB, 10dB, 15dB, clean]
# ============================================================================
RESULTS = {
    'NanoMamba-Tiny-DualPCEN': {
        'params': 4957,
        'label': 'NanoMamba (5.0K)',
        'factory': [58.6, 68.9, 79.0, 85.8, 89.0, 91.1, 91.8, 93.7],
        'white':   [37.6, 58.3, 74.7, 83.6, 88.4, 91.0, 92.3, 93.6],
        'babble':  [70.6, 77.7, 85.1, 89.3, 91.6, 92.3, 93.0, 93.8],
        'street':  [58.2, 64.6, 71.7, 83.3, 87.3, 91.0, 92.4, 93.8],
        'pink':    [28.3, 56.7, 75.4, 84.8, 89.4, 91.9, 93.2, 93.8],
    },
    'DS-CNN-S': {
        'params': 23756,
        'label': 'DS-CNN-S (23.8K)',
        'factory': [64.4, 74.6, 85.9, 91.5, 93.8, 95.1, 95.5, 96.8],
        'white':   [61.7, 67.1, 81.6, 89.9, 93.1, 94.8, 95.6, 96.8],
        'babble':  [79.2, 86.2, 91.4, 94.5, 95.5, 95.9, 96.5, 96.8],
        'street':  [61.1, 69.4, 81.5, 88.9, 92.8, 95.0, 95.9, 96.8],
        'pink':    [61.2, 66.4, 82.7, 90.6, 93.8, 95.1, 95.9, 96.8],
    },
    'BC-ResNet-1': {
        'params': 7464,
        'label': 'BC-ResNet-1 (7.5K)',
        'factory': [64.5, 74.8, 83.5, 89.0, 91.9, 92.9, 93.9, 95.3],
        'white':   [58.6, 64.6, 76.7, 85.9, 90.2, 92.6, 93.8, 95.3],
        'babble':  [76.7, 84.3, 88.8, 92.1, 93.8, 94.4, 95.0, 95.3],
        'street':  [60.7, 65.1, 77.7, 84.8, 90.1, 93.0, 94.1, 95.3],
        'pink':    [58.9, 66.0, 79.7, 88.4, 91.8, 93.8, 94.5, 95.3],
    },
}

# ============================================================================
# DualPCEN Routing Analysis Data (gate mean per noise type/SNR)
# Gate: 0=Expert1(nonstat, speech), 1=Expert2(stat, noise-robust)
# ============================================================================
ROUTING_DATA = {
    'clean':       {'mean': 0.6289, 'std': 0.0419},
    'factory_-15': {'mean': 0.6064, 'std': 0.0054},
    'factory_0':   {'mean': 0.6193, 'std': 0.0147},
    'factory_15':  {'mean': 0.6176, 'std': 0.0250},
    'white_-15':   {'mean': 0.6719, 'std': 0.0019},
    'white_0':     {'mean': 0.6806, 'std': 0.0078},
    'white_15':    {'mean': 0.6724, 'std': 0.0167},
    'babble_-15':  {'mean': 0.5219, 'std': 0.0213},
    'babble_0':    {'mean': 0.5520, 'std': 0.0316},
    'babble_15':   {'mean': 0.5920, 'std': 0.0404},
    'street_-15':  {'mean': 0.6965, 'std': 0.0017},
    'street_0':    {'mean': 0.7009, 'std': 0.0050},
    'street_15':   {'mean': 0.6802, 'std': 0.0159},
    'pink_-15':    {'mean': 0.7706, 'std': 0.0016},
    'pink_0':      {'mean': 0.7605, 'std': 0.0033},
    'pink_15':     {'mean': 0.7250, 'std': 0.0164},
}

# ============================================================================
# SS+Bypass Results: Spectral Subtraction + SNR-Adaptive Bypass (0 extra params)
# All models noise-aug trained, evaluated with SS+Bypass front-end
# Order: [-15dB, -10dB, -5dB, 0dB, 5dB, 10dB, 15dB, clean]
# ============================================================================
RESULTS_SS_BYPASS = {
    'NanoMamba-Tiny-DualPCEN': {
        'params': 4957, 'label': 'NanoMamba (5.0K)',
        'factory': [62.2, 69.6, 77.2, 83.7, 86.9, 89.9, 90.8, 93.7],
        'white':   [61.4, 68.7, 77.4, 84.1, 88.2, 90.2, 91.5, 93.6],
        'babble':  [70.5, 77.2, 83.2, 87.8, 90.5, 91.3, 92.0, 93.8],
        'street':  [59.9, 66.8, 73.1, 80.9, 85.1, 89.5, 91.1, 93.8],
        'pink':    [57.5, 70.6, 79.1, 85.5, 88.9, 91.1, 92.3, 93.8],
    },
    'DS-CNN-S': {
        'params': 23756, 'label': 'DS-CNN-S (23.8K)',
        'factory': [65.7, 73.1, 81.6, 88.1, 91.8, 94.4, 94.9, 96.8],
        'white':   [63.0, 70.3, 79.9, 86.5, 91.4, 93.7, 94.5, 96.8],
        'babble':  [79.1, 85.6, 89.9, 92.5, 93.6, 94.6, 95.2, 96.8],
        'street':  [64.0, 73.0, 79.8, 86.0, 90.8, 93.6, 94.6, 96.8],
        'pink':    [63.0, 69.7, 80.5, 87.2, 92.1, 94.2, 94.9, 96.8],
    },
    'BC-ResNet-1': {
        'params': 7464, 'label': 'BC-ResNet-1 (7.5K)',
        'factory': [66.9, 74.4, 81.2, 86.3, 90.5, 92.3, 93.5, 95.3],
        'white':   [59.1, 63.1, 69.7, 78.0, 87.5, 92.1, 93.3, 95.3],
        'babble':  [76.6, 84.0, 89.0, 92.3, 93.5, 94.1, 94.7, 95.3],
        'street':  [56.2, 63.0, 71.6, 80.2, 88.4, 92.5, 93.7, 95.3],
        'pink':    [62.0, 66.3, 76.7, 83.8, 90.2, 93.4, 94.2, 95.3],
    },
}

# ============================================================================
# Clean-Only Trained Results (from Interspeech experiments, NO noise-aug)
# Order: [-15dB, -10dB, -5dB, 0dB, 5dB, 10dB, 15dB]  (no 'clean' in noise eval)
# ============================================================================
RESULTS_CLEAN_ONLY = {
    'NanoMamba-Tiny-DualPCEN': {
        'params': 4957,
        'label': 'NanoMamba (5.0K)',
        'clean': 94.6,
        'factory': [45.2, 58.7, 70.8, 78.3, 84.0, 88.2, 91.5],
        'white':   [47.5, 59.3, 71.0, 80.5, 87.1, 91.0, 93.2],
        'babble':  [55.2, 60.8, 67.5, 74.2, 80.6, 86.5, 90.8],
    },
    'DS-CNN-S': {
        'params': 23756,
        'label': 'DS-CNN-S (23.8K)',
        'clean': 96.6,
        'factory': [59.2, 62.6, 66.4, 75.6, 83.9, 90.7, 93.3],
        'white':   [11.1, 12.0, 11.3, 13.9, 30.0, 55.6, 75.3],
        'babble':  [34.9, 45.7, 55.4, 70.1, 81.0, 88.8, 92.8],
    },
    'BC-ResNet-1': {
        'params': 7464,
        'label': 'BC-ResNet-1 (7.5K)',
        'clean': 96.0,
        'factory': [57.1, 61.5, 65.5, 71.6, 78.3, 83.8, 87.7],
        'white':   [22.0, 25.0, 37.8, 54.7, 66.1, 75.5, 84.4],
        'babble':  [37.9, 46.6, 58.0, 73.7, 85.0, 91.5, 94.1],
    },
}

SNR_LEVELS = [-15, -10, -5, 0, 5, 10, 15]
SNR_LABELS = ['-15', '-10', '-5', '0', '5', '10', '15', 'Clean']
SNR_TICKS = list(range(8))  # 0..7 for 8 data points
NOISE_TYPES = ['factory', 'white', 'babble', 'street', 'pink']
NOISE_LABELS = {'factory': 'Factory', 'white': 'White',
                'babble': 'Babble', 'street': 'Street', 'pink': 'Pink'}

# Style
MODEL_STYLES = {
    'NanoMamba-Tiny-DualPCEN': dict(color='#2563EB', marker='o', ls='-', lw=2.2, ms=7, zorder=3),
    'DS-CNN-S':               dict(color='#DC2626', marker='^', ls='--', lw=1.8, ms=7, zorder=2),
    'BC-ResNet-1':            dict(color='#16A34A', marker='s', ls='-.', lw=1.8, ms=6, zorder=2),
}

# Output dir
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
USE_PDF = '--pdf' in sys.argv
EXT = 'pdf' if USE_PDF else 'png'


def set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


# ============================================================================
# Fig A: SNR vs Accuracy — 5 Noise Types × 3 Models
# ============================================================================
def plot_snr_accuracy():
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.8), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for idx, noise in enumerate(NOISE_TYPES):
        ax = axes[idx]
        for model_name, data in RESULTS.items():
            s = MODEL_STYLES[model_name]
            ax.plot(SNR_TICKS, data[noise],
                    color=s['color'], marker=s['marker'],
                    linestyle=s['ls'], linewidth=s['lw'],
                    markersize=s['ms'], zorder=s['zorder'],
                    label=data['label'])

        ax.set_title(NOISE_LABELS[noise], fontweight='bold')
        ax.set_xticks(SNR_TICKS)
        ax.set_xticklabels(SNR_LABELS, rotation=45, ha='right')
        ax.set_xlabel('SNR (dB)')
        if idx == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(20, 100)
        ax.axhline(y=90, color='gray', ls=':', lw=0.8, alpha=0.5)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.08), frameon=True,
               fancybox=True, shadow=False)

    out = os.path.join(OUT_DIR, f'fig_snr_accuracy.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig B: Parameter Efficiency — Accuracy per 1K params
# ============================================================================
def plot_param_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # --- Left: Clean accuracy vs Params (bubble chart) ---
    ax = axes[0]
    for model_name, data in RESULTS.items():
        s = MODEL_STYLES[model_name]
        params_k = data['params'] / 1000
        # Average clean across all noise types (should be same)
        clean_acc = np.mean([data[n][-1] for n in NOISE_TYPES])
        ax.scatter(params_k, clean_acc, c=s['color'], marker=s['marker'],
                   s=150, zorder=3, edgecolors='white', linewidth=0.8,
                   label=data['label'])
        ax.annotate(f"{clean_acc:.1f}%", (params_k, clean_acc),
                    textcoords="offset points", xytext=(8, -2),
                    fontsize=8, color=s['color'])
    ax.set_xlabel('Parameters (K)')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title('(a) Clean Accuracy vs Model Size', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')

    # --- Right: Accuracy/param at different SNRs ---
    ax = axes[1]
    snr_indices = [0, 3, 7]  # -15dB, 0dB, clean
    snr_names = ['-15dB', '0dB', 'Clean']
    x = np.arange(len(snr_names))
    width = 0.25

    for i, (model_name, data) in enumerate(RESULTS.items()):
        s = MODEL_STYLES[model_name]
        params_k = data['params'] / 1000
        # Average across all noise types at each SNR level
        effs = []
        for si in snr_indices:
            avg_acc = np.mean([data[n][si] for n in NOISE_TYPES])
            effs.append(avg_acc / params_k)

        bars = ax.bar(x + (i - 1) * width, effs, width,
                       color=s['color'], alpha=0.85,
                       label=data['label'], edgecolor='white', linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, effs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=6.5,
                    color=s['color'])

    ax.set_xlabel('SNR Condition')
    ax.set_ylabel('Accuracy / K-params (%/K)')
    ax.set_title('(b) Parameter Efficiency', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_names)
    ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_param_efficiency.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig C: Accuracy Drop Heatmap (Clean → 0dB)
# ============================================================================
def plot_accuracy_drop_heatmap():
    fig, ax = plt.subplots(figsize=(5.5, 2.2))

    model_names = list(RESULTS.keys())
    model_labels = [RESULTS[m]['label'] for m in model_names]
    n_models = len(model_names)
    n_noise = len(NOISE_TYPES)

    # Compute drops: clean_acc - 0dB_acc
    drop_matrix = np.zeros((n_models, n_noise))
    for i, model_name in enumerate(model_names):
        for j, noise in enumerate(NOISE_TYPES):
            clean = RESULTS[model_name][noise][-1]   # last = clean
            zero_db = RESULTS[model_name][noise][3]   # index 3 = 0dB
            drop_matrix[i, j] = clean - zero_db

    im = ax.imshow(drop_matrix, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=15)

    # Labels
    ax.set_xticks(range(n_noise))
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_TYPES])
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels)

    # Annotate values
    for i in range(n_models):
        for j in range(n_noise):
            val = drop_matrix[i, j]
            color = 'white' if val > 10 else 'black'
            ax.text(j, i, f'{val:.1f}%p', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_title('Accuracy Drop: Clean → 0 dB (lower is better)', fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Drop (%p)', fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_accuracy_drop.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig D: Radar/Spider Chart — 0dB performance across noise types
# ============================================================================
def plot_radar():
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    categories = [NOISE_LABELS[n] for n in NOISE_TYPES]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for model_name, data in RESULTS.items():
        s = MODEL_STYLES[model_name]
        # 0dB values (index 3)
        values = [data[n][3] for n in NOISE_TYPES]
        values += values[:1]  # close
        ax.plot(angles, values, color=s['color'], linewidth=s['lw'],
                linestyle=s['ls'], marker=s['marker'], markersize=5,
                label=data['label'])
        ax.fill(angles, values, color=s['color'], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(75, 96)
    ax.set_yticks([80, 85, 90, 95])
    ax.set_yticklabels(['80%', '85%', '90%', '95%'], fontsize=7, color='gray')
    ax.set_title('Accuracy at 0 dB SNR', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=7.5)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_radar_0dB.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig E: Extreme SNR (-15, -10, -5 dB) Average — Bar Chart
# ============================================================================
def plot_extreme_bar():
    fig, ax = plt.subplots(figsize=(6, 3))

    x = np.arange(len(NOISE_TYPES))
    width = 0.25

    for i, (model_name, data) in enumerate(RESULTS.items()):
        s = MODEL_STYLES[model_name]
        # Average of -15, -10, -5 dB (indices 0, 1, 2)
        extreme_avgs = [np.mean(data[n][:3]) for n in NOISE_TYPES]
        bars = ax.bar(x + (i - 1) * width, extreme_avgs, width,
                       color=s['color'], alpha=0.85,
                       label=data['label'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, extreme_avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7,
                    color=s['color'], fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Average Accuracy at Extreme SNR (-15, -10, -5 dB)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_TYPES])
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_ylim(0, 95)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_extreme_snr.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig F: Individual Noise Type Plots (large, detailed)
# ============================================================================
def plot_per_noise():
    """Generate one detailed figure per noise type with data annotations."""
    for noise in NOISE_TYPES:
        fig, ax = plt.subplots(figsize=(6, 4))

        for model_name, data in RESULTS.items():
            s = MODEL_STYLES[model_name]
            vals = data[noise]
            ax.plot(SNR_TICKS, vals,
                    color=s['color'], marker=s['marker'],
                    linestyle=s['ls'], linewidth=s['lw'] + 0.3,
                    markersize=s['ms'] + 1, zorder=s['zorder'],
                    label=data['label'])
            # Annotate each point
            for xi, yi in zip(SNR_TICKS, vals):
                offset_y = 2.5 if model_name == 'NanoMamba-Tiny-DualPCEN' else -4.5
                if model_name == 'BC-ResNet-1':
                    offset_y = -4.5 if yi < data[noise][-1] else 2.5
                ax.annotate(f'{yi:.1f}', (xi, yi),
                            textcoords="offset points",
                            xytext=(0, offset_y),
                            ha='center', fontsize=7,
                            color=s['color'], fontweight='bold')

        # Highlight zones
        ax.axhspan(90, 100, color='green', alpha=0.04)
        ax.axhspan(0, 70, color='red', alpha=0.04)
        ax.axhline(y=90, color='gray', ls=':', lw=0.8, alpha=0.5)

        # Gap annotation at 0dB
        nm_0db = RESULTS['NanoMamba-Tiny-DualPCEN'][noise][3]
        ds_0db = RESULTS['DS-CNN-S'][noise][3]
        gap = ds_0db - nm_0db
        ax.annotate(f'Gap: {gap:.1f}%p',
                    xy=(3, (nm_0db + ds_0db) / 2),
                    fontsize=8, color='#666666', ha='left',
                    xytext=(3.4, (nm_0db + ds_0db) / 2),
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.8))

        ax.set_title(f'{NOISE_LABELS[noise]} Noise - SNR vs Accuracy',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(SNR_TICKS)
        ax.set_xticklabels(SNR_LABELS, fontsize=9)
        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_ylim(15 if noise in ['white', 'pink'] else 50, 100)
        ax.legend(fontsize=9, loc='lower right',
                  framealpha=0.9, edgecolor='gray')

        # Parameter info box
        info = "Params: NanoMamba 5.0K | DS-CNN-S 23.8K | BC-ResNet-1 7.5K"
        ax.text(0.5, 0.02, info, transform=ax.transAxes,
                fontsize=7, ha='center', color='gray', style='italic')

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'fig_noise_{noise}.{EXT}')
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")


# ============================================================================
# Fig G: Clean-Only vs Noise-Aug Comparison (KEY EVIDENCE)
# ============================================================================
def plot_clean_vs_noiseaug():
    """Compare clean-only vs noise-aug trained models at 0dB — the smoking gun."""
    noise_subset = ['factory', 'white', 'babble']
    noise_labels_sub = ['Factory', 'White', 'Babble']
    models = ['NanoMamba-Tiny-DualPCEN', 'DS-CNN-S', 'BC-ResNet-1']
    short_labels = ['NanoMamba\n(5.0K)', 'DS-CNN-S\n(23.8K)', 'BC-ResNet-1\n(7.5K)']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    colors_clean = ['#93C5FD', '#FCA5A5', '#86EFAC']   # light
    colors_aug   = ['#2563EB', '#DC2626', '#16A34A']    # dark

    for idx, noise in enumerate(noise_subset):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35

        # Clean-only trained (0dB = index 3)
        clean_vals = [RESULTS_CLEAN_ONLY[m][noise][3] for m in models]
        # Noise-aug trained (0dB = index 3)
        aug_vals = [RESULTS[m][noise][3] for m in models]

        bars1 = ax.bar(x - width/2, clean_vals, width,
                        color=colors_clean, edgecolor='white', linewidth=0.8,
                        label='Clean-only trained' if idx == 0 else '')
        bars2 = ax.bar(x + width/2, aug_vals, width,
                        color=colors_aug, edgecolor='white', linewidth=0.8,
                        label='Noise-aug trained' if idx == 0 else '')

        # Value + delta annotations
        for i, (cv, av) in enumerate(zip(clean_vals, aug_vals)):
            ax.text(x[i] - width/2, cv + 1.0, f'{cv:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color='#555555')
            ax.text(x[i] + width/2, av + 1.0, f'{av:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=colors_aug[i])
            # Delta arrow
            delta = av - cv
            ax.annotate(f'+{delta:.1f}',
                        xy=(x[i] + width/2, av - 1),
                        fontsize=7, color='#333333', ha='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='#FEF3C7', edgecolor='#F59E0B',
                                  alpha=0.9))

        ax.set_title(f'{noise_labels_sub[idx]} Noise (0 dB)',
                     fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8)
        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_ylim(0, 102)
        ax.axhline(y=90, color='gray', ls=':', lw=0.8, alpha=0.4)

        # Highlight NanoMamba's structural advantage in clean-only
        if noise == 'white':
            ax.annotate('Structural\ncollapse!',
                        xy=(1, clean_vals[1] + 2), fontsize=7,
                        color='#DC2626', ha='center', fontweight='bold')

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc='#AAAAAA', ec='white'),
               plt.Rectangle((0, 0), 1, 1, fc='#555555', ec='white')]
    fig.legend(handles, ['Clean-only trained', 'Noise-aug trained'],
               loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.06), fontsize=9,
               frameon=True, fancybox=True)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_clean_vs_noiseaug.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig H: Structural Robustness Summary (avg across noise types)
# ============================================================================
def plot_structural_summary():
    """Bar chart: average 0dB accuracy clean-only vs noise-aug."""
    models = ['NanoMamba-Tiny-DualPCEN', 'DS-CNN-S', 'BC-ResNet-1']
    short_labels = ['NanoMamba\n(5.0K)', 'DS-CNN-S\n(23.8K)', 'BC-ResNet-1\n(7.5K)']
    noise_subset = ['factory', 'white', 'babble']

    # Average 0dB across 3 noise types
    clean_avgs = []
    aug_avgs = []
    for m in models:
        cavg = np.mean([RESULTS_CLEAN_ONLY[m][n][3] for n in noise_subset])
        aavg = np.mean([RESULTS[m][n][3] for n in noise_subset])
        clean_avgs.append(cavg)
        aug_avgs.append(aavg)

    deltas = [a - c for c, a in zip(clean_avgs, aug_avgs)]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    width = 0.32

    colors_main = ['#2563EB', '#DC2626', '#16A34A']

    # Clean-only bars
    bars1 = ax.bar(x - width/2, clean_avgs, width,
                    color=[c + '55' for c in ['#2563EB', '#DC2626', '#16A34A']],
                    edgecolor=colors_main, linewidth=1.5,
                    label='Clean-only trained (structural)')
    # Noise-aug bars
    bars2 = ax.bar(x + width/2, aug_avgs, width,
                    color=colors_main, alpha=0.85,
                    edgecolor='white', linewidth=0.8,
                    label='Noise-aug trained (structural + training)')

    for i in range(len(models)):
        ax.text(x[i] - width/2, clean_avgs[i] + 1, f'{clean_avgs[i]:.1f}%',
                ha='center', fontsize=9, fontweight='bold', color=colors_main[i])
        ax.text(x[i] + width/2, aug_avgs[i] + 1, f'{aug_avgs[i]:.1f}%',
                ha='center', fontsize=9, fontweight='bold', color=colors_main[i])
        # Delta
        ax.annotate(f'+{deltas[i]:.1f}%p',
                    xy=(x[i], (clean_avgs[i] + aug_avgs[i]) / 2),
                    fontsize=9, ha='center', fontweight='bold',
                    color='#B45309',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#FEF3C7', edgecolor='#F59E0B'))

    ax.set_ylabel('Average Accuracy at 0 dB (%)', fontsize=10)
    ax.set_title('Structural vs Training-Based Noise Robustness\n'
                 '(Average of Factory, White, Babble at 0 dB)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8, loc='upper right')
    ax.axhline(y=90, color='gray', ls=':', lw=0.8, alpha=0.4)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_structural_summary.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Fig I: Clean-Only Trained — Per Noise Type (large, detailed)
# ============================================================================
def plot_per_noise_clean_only():
    """Individual noise plots for clean-only trained models (no noise-aug)."""
    clean_noise_types = ['factory', 'white', 'babble']
    clean_noise_labels = {'factory': 'Factory', 'white': 'White (Broadband)',
                          'babble': 'Babble (Non-stationary)'}
    snr_ticks_co = list(range(7))  # 0..6 for 7 data points
    snr_labels_co = ['-15', '-10', '-5', '0', '5', '10', '15']

    for noise in clean_noise_types:
        fig, ax = plt.subplots(figsize=(6, 4))

        for model_name, data in RESULTS_CLEAN_ONLY.items():
            s = MODEL_STYLES[model_name]
            vals = data[noise]
            ax.plot(snr_ticks_co, vals,
                    color=s['color'], marker=s['marker'],
                    linestyle=s['ls'], linewidth=s['lw'] + 0.3,
                    markersize=s['ms'] + 1, zorder=s['zorder'],
                    label=data['label'])
            # Annotate each point
            for xi, yi in zip(snr_ticks_co, vals):
                if model_name == 'NanoMamba-Tiny-DualPCEN':
                    offset_y = 3.5
                elif model_name == 'DS-CNN-S':
                    offset_y = -5.5
                else:
                    offset_y = -5.5 if yi < vals[-1] * 0.95 else 3.5
                ax.annotate(f'{yi:.1f}', (xi, yi),
                            textcoords="offset points",
                            xytext=(0, offset_y),
                            ha='center', fontsize=7.5,
                            color=s['color'], fontweight='bold')

        # Highlight zones
        ax.axhspan(90, 100, color='green', alpha=0.04)
        ax.axhspan(0, 50, color='red', alpha=0.05)
        ax.axhline(y=90, color='gray', ls=':', lw=0.8, alpha=0.5)

        # Gap annotation at 0dB (index 3)
        nm_0db = RESULTS_CLEAN_ONLY['NanoMamba-Tiny-DualPCEN'][noise][3]
        ds_0db = RESULTS_CLEAN_ONLY['DS-CNN-S'][noise][3]
        gap = nm_0db - ds_0db  # NanoMamba wins in clean-only!
        gap_color = '#2563EB' if gap > 0 else '#DC2626'
        gap_text = f'NanoMamba +{gap:.1f}%p' if gap > 0 else f'DS-CNN-S +{-gap:.1f}%p'
        ax.annotate(gap_text,
                    xy=(3, (nm_0db + ds_0db) / 2),
                    fontsize=8, color=gap_color, ha='left',
                    fontweight='bold',
                    xytext=(3.5, (nm_0db + ds_0db) / 2 + 3),
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#EFF6FF', edgecolor=gap_color, alpha=0.9))

        # White noise: highlight DS-CNN-S collapse
        if noise == 'white':
            ax.annotate('DS-CNN-S COLLAPSE\n(near random)',
                        xy=(3, ds_0db), fontsize=8,
                        color='#DC2626', ha='center', fontweight='bold',
                        xytext=(4.5, 25),
                        arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5))

        ax.set_title(f'{clean_noise_labels[noise]} Noise - Clean-Only Trained\n(NO noise augmentation)',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(snr_ticks_co)
        ax.set_xticklabels(snr_labels_co, fontsize=9)
        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        y_min = 5 if noise == 'white' else 25
        ax.set_ylim(y_min, 100)
        ax.legend(fontsize=9, loc='lower right',
                  framealpha=0.9, edgecolor='gray')

        info = "Clean-only trained (structural robustness only, NO noise in training)"
        ax.text(0.5, 0.02, info, transform=ax.transAxes,
                fontsize=7, ha='center', color='#B45309', style='italic',
                fontweight='bold')

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'fig_cleanonly_{noise}.{EXT}')
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")


# ============================================================================
# Fig J: DualPCEN Routing Analysis Heatmap
# ============================================================================
def plot_routing_analysis():
    """Heatmap of DualPCEN gate values across noise types and SNR levels.
    Shows how the routing adapts: babble→Expert1 (nonstat), pink→Expert2 (stat)."""
    noise_types_r = ['factory', 'white', 'babble', 'street', 'pink']
    snr_levels_r = [-15, 0, 15]
    snr_labels_r = ['-15 dB', '0 dB', '15 dB']

    # Build heatmap matrix (noise × SNR)
    n_noise = len(noise_types_r)
    n_snr = len(snr_levels_r) + 1  # +1 for clean
    gate_matrix = np.zeros((n_noise, n_snr))
    std_matrix = np.zeros((n_noise, n_snr))

    for i, noise in enumerate(noise_types_r):
        for j, snr in enumerate(snr_levels_r):
            key = f'{noise}_{snr}'
            gate_matrix[i, j] = ROUTING_DATA[key]['mean']
            std_matrix[i, j] = ROUTING_DATA[key]['std']
        # Clean column (same for all noise types)
        gate_matrix[i, n_snr - 1] = ROUTING_DATA['clean']['mean']
        std_matrix[i, n_snr - 1] = ROUTING_DATA['clean']['std']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                                     gridspec_kw={'width_ratios': [3, 2]})

    # --- Left: Heatmap ---
    col_labels = snr_labels_r + ['Clean']
    row_labels = [NOISE_LABELS[n] for n in noise_types_r]

    im = ax1.imshow(gate_matrix, cmap='RdYlBu_r', aspect='auto',
                     vmin=0.48, vmax=0.80)

    ax1.set_xticks(range(n_snr))
    ax1.set_xticklabels(col_labels, fontsize=9)
    ax1.set_yticks(range(n_noise))
    ax1.set_yticklabels(row_labels, fontsize=9)

    # Annotate values
    for i in range(n_noise):
        for j in range(n_snr):
            val = gate_matrix[i, j]
            std = std_matrix[i, j]
            color = 'white' if val > 0.70 or val < 0.55 else 'black'
            ax1.text(j, i, f'{val:.3f}\n({std:.3f})',
                     ha='center', va='center', fontsize=7.5,
                     fontweight='bold', color=color)

    ax1.set_title('DualPCEN Gate Value (0=Expert1/nonstat, 1=Expert2/stat)',
                   fontweight='bold', fontsize=10)
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label('Gate Mean', fontsize=8)

    # Add arrows for interpretation
    ax1.annotate('More Expert2\n(noise-robust)',
                 xy=(0, 4), xytext=(-0.8, 4),
                 fontsize=7, color='#DC2626', fontweight='bold',
                 ha='right', va='center')
    ax1.annotate('More Expert1\n(speech-focused)',
                 xy=(0, 2), xytext=(-0.8, 2),
                 fontsize=7, color='#2563EB', fontweight='bold',
                 ha='right', va='center')

    # --- Right: Bar chart of gate separation from clean ---
    separations = {}
    for noise in noise_types_r:
        key_m15 = f'{noise}_-15'
        sep = ROUTING_DATA[key_m15]['mean'] - ROUTING_DATA['clean']['mean']
        separations[noise] = sep

    noise_labels_list = [NOISE_LABELS[n] for n in noise_types_r]
    sep_vals = [separations[n] for n in noise_types_r]
    colors_bar = ['#DC2626' if v > 0 else '#2563EB' for v in sep_vals]

    bars = ax2.barh(range(n_noise), sep_vals, color=colors_bar, alpha=0.8,
                     edgecolor='white', linewidth=0.8)
    ax2.set_yticks(range(n_noise))
    ax2.set_yticklabels(noise_labels_list, fontsize=9)
    ax2.axvline(x=0, color='gray', lw=1, ls='-')
    ax2.set_xlabel('Gate Separation from Clean', fontsize=9)
    ax2.set_title('Routing Shift at -15dB\n(vs Clean baseline)',
                   fontweight='bold', fontsize=10)

    for i, (bar, val) in enumerate(zip(bars, sep_vals)):
        sign = '+' if val > 0 else ''
        x_pos = val + 0.005 if val > 0 else val - 0.005
        ha = 'left' if val > 0 else 'right'
        ax2.text(x_pos, i, f'{sign}{val:.3f}', va='center', ha=ha,
                 fontsize=8, fontweight='bold', color=colors_bar[i])

    # Legend annotation
    ax2.text(0.95, 0.05,
             'Red: → Expert2 (stat/AGC)\nBlue: → Expert1 (nonstat/speech)',
             transform=ax2.transAxes, fontsize=7,
             va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F9FAFB',
                       edgecolor='#D1D5DB'))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_routing_analysis.{EXT}')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# SS+Bypass: Baseline vs Enhanced Comparison
# ============================================================================

def plot_ss_bypass_comparison():
    """Fig: SS+Bypass improvement across all models and noise types."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    noises = NOISE_TYPES
    labels = [NOISE_LABELS[n] for n in noises]
    models_list = ['NanoMamba-Tiny-DualPCEN', 'DS-CNN-S', 'BC-ResNet-1']
    colors = ['#2563EB', '#DC2626', '#16A34A']
    short_labels = ['NanoMamba', 'DS-CNN-S', 'BC-ResNet-1']

    for panel_idx, (snr_idx, snr_label, title) in enumerate([
        (0, '-15dB', '(a) Improvement at -15 dB SNR'),
        (3, '0dB', '(b) Change at 0 dB SNR'),
    ]):
        ax = axes[panel_idx]
        x = np.arange(len(noises))
        w = 0.22

        for i, (mname, color, slabel) in enumerate(zip(models_list, colors, short_labels)):
            deltas = []
            for noise in noises:
                base = RESULTS[mname][noise][snr_idx]
                ss = RESULTS_SS_BYPASS[mname][noise][snr_idx]
                deltas.append(ss - base)
            bars = ax.bar(x + (i - 1) * w, deltas, w * 0.9,
                         color=color, alpha=0.85, label=slabel,
                         edgecolor='white', linewidth=0.5)
            # Annotate values
            for j, (bar, d) in enumerate(zip(bars, deltas)):
                va = 'bottom' if d >= 0 else 'top'
                offset = 0.3 if d >= 0 else -0.3
                ax.text(bar.get_x() + bar.get_width() / 2, d + offset,
                       f'{d:+.1f}', ha='center', va=va, fontsize=6.5,
                       fontweight='bold', color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Accuracy Change (%p)')
        ax.set_title(title, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax.grid(axis='y', alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('Spectral Subtraction + SNR-Adaptive Bypass (0 extra params)',
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_ss_bypass.{EXT}')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_ss_bypass_snr_curves():
    """Fig: Full SNR curves — baseline (dashed) vs SS+Bypass (solid) for NanoMamba."""
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)

    nm = 'NanoMamba-Tiny-DualPCEN'
    bc = 'BC-ResNet-1'

    for i, noise in enumerate(NOISE_TYPES):
        ax = axes[i]
        # NanoMamba baseline (dashed)
        ax.plot(SNR_TICKS, RESULTS[nm][noise],
                color='#2563EB', ls='--', lw=1.2, alpha=0.5,
                marker='o', ms=3, label='NM base')
        # NanoMamba SS+Bypass (solid)
        ax.plot(SNR_TICKS, RESULTS_SS_BYPASS[nm][noise],
                color='#2563EB', ls='-', lw=2, marker='o', ms=4,
                label='NM +SS')
        # BC-ResNet-1 baseline (dashed gray)
        ax.plot(SNR_TICKS, RESULTS[bc][noise],
                color='#16A34A', ls='--', lw=1.2, alpha=0.5,
                marker='s', ms=3, label='BC base')
        # BC-ResNet-1 SS+Bypass (solid)
        ax.plot(SNR_TICKS, RESULTS_SS_BYPASS[bc][noise],
                color='#16A34A', ls='-', lw=1.5, marker='s', ms=3,
                label='BC +SS')

        # Shade improvement region for NanoMamba
        base_arr = np.array(RESULTS[nm][noise])
        ss_arr = np.array(RESULTS_SS_BYPASS[nm][noise])
        ax.fill_between(SNR_TICKS, base_arr, ss_arr,
                        where=ss_arr > base_arr,
                        alpha=0.15, color='#2563EB')

        ax.set_xticks(SNR_TICKS)
        ax.set_xticklabels(SNR_LABELS, fontsize=7)
        ax.set_title(NOISE_LABELS[noise], fontweight='bold', fontsize=9)
        ax.set_ylim(20, 100)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel('Accuracy (%)')
            ax.legend(fontsize=6, loc='lower right')

    fig.suptitle('NanoMamba (5K) vs BC-ResNet-1 (7.5K): Baseline vs +SS+Bypass',
                 fontsize=10, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_ss_bypass_curves.{EXT}')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_ss_bypass_wins():
    """Fig: Where NanoMamba (5K) beats BC-ResNet-1 (7.5K) with SS+Bypass."""
    fig, ax = plt.subplots(figsize=(8, 4))

    noises = NOISE_TYPES
    nm = 'NanoMamba-Tiny-DualPCEN'
    bc = 'BC-ResNet-1'

    # Compute extreme SNR average (-15, -10, -5 dB) = indices 0,1,2
    nm_extreme = []
    bc_extreme = []
    for noise in noises:
        nm_avg = np.mean(RESULTS_SS_BYPASS[nm][noise][:3])
        bc_avg = np.mean(RESULTS_SS_BYPASS[bc][noise][:3])
        nm_extreme.append(nm_avg)
        bc_extreme.append(bc_avg)

    x = np.arange(len(noises))
    w = 0.3
    bars_nm = ax.bar(x - w/2, nm_extreme, w, color='#2563EB', alpha=0.85,
                     label=f'NanoMamba (5.0K)', edgecolor='white')
    bars_bc = ax.bar(x + w/2, bc_extreme, w, color='#16A34A', alpha=0.85,
                     label=f'BC-ResNet-1 (7.5K)', edgecolor='white')

    # Annotate with delta and winner
    for i in range(len(noises)):
        delta = nm_extreme[i] - bc_extreme[i]
        winner = delta > 0
        y_pos = max(nm_extreme[i], bc_extreme[i]) + 1.5
        color = '#2563EB' if winner else '#16A34A'
        symbol = 'NM WINS' if winner else 'BC wins'
        ax.text(x[i], y_pos, f'{delta:+.1f}%p\n{symbol}',
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               color=color)
        # Values on bars
        ax.text(x[i] - w/2, nm_extreme[i] + 0.3, f'{nm_extreme[i]:.1f}',
               ha='center', va='bottom', fontsize=7, color='#2563EB')
        ax.text(x[i] + w/2, bc_extreme[i] + 0.3, f'{bc_extreme[i]:.1f}',
               ha='center', va='bottom', fontsize=7, color='#16A34A')

    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS[n] for n in noises])
    ax.set_ylabel('Avg Accuracy at Extreme SNR (-15,-10,-5 dB)')
    ax.set_title('SS+Bypass: NanoMamba (5K) vs BC-ResNet-1 (7.5K) at Extreme Noise',
                fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(50, 90)
    ax.grid(axis='y', alpha=0.3)

    # Summary text
    wins = sum(1 for i in range(len(noises)) if nm_extreme[i] > bc_extreme[i])
    ax.text(0.02, 0.02,
           f'NanoMamba wins {wins}/{len(noises)} noise types\n'
           f'with 1.5x FEWER params & 10x fewer MACs',
           transform=ax.transAxes, fontsize=8,
           va='bottom', ha='left',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#EFF6FF',
                     edgecolor='#2563EB', alpha=0.9))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'fig_ss_bypass_wins.{EXT}')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    set_style()
    print(f"\nGenerating NanoMamba result figures ({EXT.upper()})...\n")

    plot_snr_accuracy()
    plot_param_efficiency()
    plot_accuracy_drop_heatmap()
    plot_radar()
    plot_extreme_bar()
    plot_per_noise()
    plot_clean_vs_noiseaug()
    plot_structural_summary()
    plot_per_noise_clean_only()
    plot_routing_analysis()
    plot_ss_bypass_comparison()
    plot_ss_bypass_snr_curves()
    plot_ss_bypass_wins()

    print(f"\nDone! All figures saved to: {OUT_DIR}/")
    print(f"\nKey observations:")
    print(f"  - DS-CNN-S leads in absolute accuracy (4.8x more params)")
    print(f"  - NanoMamba achieves 4.6x better parameter efficiency")
    print(f"  - White/Pink noise: largest gap (broadband, no spectral structure)")
    print(f"  - Babble noise: smallest gap (DualPCEN routing helps)")
    print(f"  - Routing: Pink/Street→Expert2(stat), Babble→Expert1(nonstat)")
