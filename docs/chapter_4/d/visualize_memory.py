"""
Thực nghiệm (d): Sử dụng bộ nhớ (Peak Memory)
Tách ra 2 hình riêng:
  1. Grouped bar chart (Basic vs Optimized)
  2. Tỉ lệ nén bộ nhớ (Basic / Optimized)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

script_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(script_dir, "memory.csv"))

# ====== Hình 1: Grouped bar chart (Basic vs Optimized) ======
fig1, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(df))
width = 0.35

bars_basic = ax1.bar(x - width/2, df['mem_basic_mb'], width,
                     label='Basic', color='#4C72B0', edgecolor='white', linewidth=0.5)
bars_opt = ax1.bar(x + width/2, df['mem_opt_mb'], width,
                   label='Optimized', color='#55A868', edgecolor='white', linewidth=0.5)

ax1.set_xlabel('Tập dữ liệu')
ax1.set_ylabel('Peak Memory (MB)')
ax1.set_title('(d-1) Peak Memory: Basic vs Optimized', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
labels = [f"{row['dataset']}\n(minsup={row['minsup_pct']}%)" for _, row in df.iterrows()]
ax1.set_xticklabels(labels, fontsize=10)
ax1.legend(fontsize=10)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars_basic, bars_opt]:
    for bar in bars:
        height = bar.get_height()
        if height > 1000:
            txt = f'{height/1024:.1f} GB'
        else:
            txt = f'{height:.1f} MB'
        ax1.annotate(txt,
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, rotation=45)

fig1.tight_layout()
fig1.savefig(os.path.join(script_dir, "d1_peak_memory.png"), dpi=200, bbox_inches='tight')
print("Saved: d1_peak_memory.png")

# ====== Hình 2: Tỉ lệ nén (Basic / Optimized) ======
fig2, ax2 = plt.subplots(figsize=(10, 6))
ratios = df['mem_basic_mb'] / df['mem_opt_mb']
colors = ['#2ecc71' if r > 1 else '#e74c3c' for r in ratios]

bars_ratio = ax2.barh(df['dataset'], ratios, color=colors, edgecolor='white', height=0.5)
ax2.axvline(x=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (1x)')
ax2.set_xlabel('Tỉ lệ nén bộ nhớ (Basic / Optimized)')
ax2.set_title('(d-2) Tỉ lệ nén bộ nhớ Basic so với Optimized', fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

for bar, ratio in zip(bars_ratio, ratios):
    ax2.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height()/2,
             f'{ratio:,.4f}x', ha='left', va='center', fontsize=11, fontweight='bold',
             color='#2ecc71' if ratio > 1 else '#e74c3c')

fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, "d2_memory_ratio.png"), dpi=200, bbox_inches='tight')
print("Saved: d2_memory_ratio.png")

plt.show()

# --- Summary ---
print("\n" + "="*80)
print("BANG TOM TAT SU DUNG BO NHO")
print("="*80)
for _, row in df.iterrows():
    ratio = row['mem_basic_mb'] / row['mem_opt_mb']
    print(f"  {row['dataset']:15s} | minsup={row['minsup_pct']:5.1f}% | "
          f"Basic={row['mem_basic_mb']:>10.2f} MB | Opt={row['mem_opt_mb']:>12.2f} MB | "
          f"Ratio={ratio:>8,.0f}x")
print("="*80)
