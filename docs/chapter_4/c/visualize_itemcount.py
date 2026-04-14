"""
Thực nghiệm (c): Số lượng Frequent Itemset theo minsup
Tách ra 2 hình riêng:
  1. Tất cả datasets trên cùng 1 đồ thị (log scale)
  2. Tốc độ tăng trưởng FI khi minsup giảm
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

script_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(script_dir, "itemcount_vs_minsup.csv"))
datasets = df['dataset'].unique()

ds_info = {
    'Chess':       {'type': 'Dense',  'color': '#4C72B0', 'marker': 'o'},
    'Mushroom':    {'type': 'Dense',  'color': '#55A868', 'marker': 's'},
    'Retail':      {'type': 'Sparse', 'color': '#DD8452', 'marker': '^'},
    'Accidents':   {'type': 'Dense',  'color': '#C44E52', 'marker': 'D'},
    'T10I4D100K':  {'type': 'Sparse', 'color': '#8172B3', 'marker': 'v'},
}

# ====== Hình 1: Tất cả datasets (log scale) ======
fig1, ax1 = plt.subplots(figsize=(10, 7))
for dataset in datasets:
    subset = df[df['dataset'] == dataset].sort_values('minsup_pct', ascending=False)
    info = ds_info.get(dataset, {'type': '?', 'color': 'gray', 'marker': 'o'})
    linestyle = '-' if info['type'] == 'Dense' else '--'
    ax1.plot(subset['minsup_pct'], subset['n_fi'],
             f'{info["marker"]}{linestyle}', color=info['color'],
             linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=0.5,
             label=f'{dataset} ({info["type"]})')

ax1.set_xlabel('minsup (%)')
ax1.set_ylabel('Số lượng Frequent Itemset')
ax1.set_title('(c-1) Số lượng FI theo minsup — Tất cả tập dữ liệu', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.invert_xaxis()

fig1.tight_layout()
fig1.savefig(os.path.join(script_dir, "c1_fi_count_all.png"), dpi=200, bbox_inches='tight')
print("Saved: c1_fi_count_all.png")

# ====== Hình 2: Tốc độ tăng trưởng (growth ratio) ======
fig2, ax2 = plt.subplots(figsize=(10, 7))
for dataset in datasets:
    subset = df[df['dataset'] == dataset].sort_values('minsup_pct', ascending=False)
    n_fi = subset['n_fi'].values
    minsup = subset['minsup_pct'].values

    base_val = max(n_fi[0], 1)
    growth = n_fi / base_val

    info = ds_info.get(dataset, {'type': '?', 'color': 'gray', 'marker': 'o'})
    linestyle = '-' if info['type'] == 'Dense' else '--'
    ax2.plot(minsup, growth,
             f'{info["marker"]}{linestyle}', color=info['color'],
             linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=0.5,
             label=f'{dataset} ({info["type"]})')

ax2.set_xlabel('minsup (%)')
ax2.set_ylabel('Tỉ lệ tăng trưởng (so với minsup cao nhất)')
ax2.set_title('(c-2) Tốc độ tăng trưởng FI khi minsup giảm', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.invert_xaxis()

fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, "c2_fi_growth_rate.png"), dpi=200, bbox_inches='tight')
print("Saved: c2_fi_growth_rate.png")

plt.show()

# --- Summary ---
print("\n" + "="*80)
print("NHẬN XÉT: MỐI QUAN HỆ GIỮA MINSUP VÀ OUTPUT SIZE")
print("="*80)
for dataset in datasets:
    subset = df[df['dataset'] == dataset].sort_values('minsup_pct')
    info = ds_info.get(dataset, {'type': '?'})
    min_row = subset.iloc[0]
    max_row = subset.iloc[-1]
    max_fi = max(min_row['n_fi'], 1)
    min_fi = max(max_row['n_fi'], 1)
    ratio = max_fi / min_fi
    print(f"  {dataset:15s} ({info['type']:6s}) | "
          f"minsup {max_row['minsup_pct']:.1f}% -> {min_row['minsup_pct']:.1f}%: "
          f"FI tang tu {int(max_row['n_fi']):>8,} -> {int(min_row['n_fi']):>8,} "
          f"(x{ratio:,.0f})")
print("="*80)
print("\n-> Tap du lieu DENSE (Chess, Mushroom, Accidents): FI bung no nhanh khi minsup giam")
print("-> Tap du lieu SPARSE (Retail, T10I4D100K): FI tang cham hon, output nho hon nhieu")
