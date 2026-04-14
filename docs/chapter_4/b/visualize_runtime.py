"""
Thực nghiệm (b): Thời gian chạy theo minsup
Tách ra 1 hình riêng cho mỗi tập dữ liệu (Our vs SPMF)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

script_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(script_dir, "runtime_vs_minsup.csv"))
datasets = df['dataset'].unique()

colors_ours = '#4C72B0'
colors_spmf = '#C44E52'

for dataset in datasets:
    subset = df[df['dataset'] == dataset].sort_values('minsup_pct', ascending=False)

    minsup_vals = subset['minsup_pct'].values
    time_ours = subset['time_ours_ms'].values
    time_spmf = subset['time_spmf_ms'].values

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(minsup_vals, time_ours, 'o-', color=colors_ours, linewidth=2, markersize=7,
            label='Our Implementation', markeredgecolor='white', markeredgewidth=0.5)
    ax.plot(minsup_vals, time_spmf, 's-', color=colors_spmf, linewidth=2, markersize=7,
            label='SPMF', markeredgecolor='white', markeredgewidth=0.5)

    ax.fill_between(minsup_vals, time_ours, time_spmf, alpha=0.1, color='gray')

    ax.set_title(f'(b) Thời gian chạy theo minsup — {dataset}', fontsize=14, fontweight='bold')
    ax.set_xlabel('minsup (%)')
    ax.set_ylabel('Thời gian (ms)')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.invert_xaxis()

    # Annotate speedup ratio at lowest minsup
    ratio = time_ours[-1] / time_spmf[-1]
    if ratio > 1:
        label = f'Chậm hơn {ratio:.0f}x'
        color = '#e74c3c'
    else:
        label = f'Nhanh hơn {1/ratio:.1f}x'
        color = '#2ecc71'
    ax.annotate(label, xy=(0.02, 0.92), xycoords='axes fraction',
                fontsize=11, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

    fig.tight_layout()
    fname = f"b_runtime_{dataset.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(script_dir, fname), dpi=200, bbox_inches='tight')
    print(f"Saved: {fname}")

plt.show()

# --- Summary ---
print("\n" + "="*80)
print("TÓM TẮT THỜI GIAN CHẠY (tại minsup thấp nhất)")
print("="*80)
for dataset in datasets:
    subset = df[df['dataset'] == dataset].sort_values('minsup_pct')
    row = subset.iloc[0]
    ratio = row['time_ours_ms'] / row['time_spmf_ms']
    print(f"  {dataset:15s} | minsup={row['minsup_pct']:5.1f}% | "
          f"Ours={row['time_ours_ms']:>12,.2f} ms | SPMF={row['time_spmf_ms']:>10,.2f} ms | "
          f"Ratio={ratio:,.1f}x")
print("="*80)
