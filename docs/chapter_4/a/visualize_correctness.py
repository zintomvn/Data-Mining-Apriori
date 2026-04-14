"""
Thực nghiệm (a): Kiểm tra tính đúng đắn (Correctness)
Tách ra 3 hình riêng biệt:
  1. So sánh số lượng frequent itemset (Basic / Opt / SPMF)
  2. Tỉ lệ khớp (match %)
  3. So sánh thời gian chạy (Our vs SPMF)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "correctness.csv"))

# ====== Hình 1: So sánh số lượng Frequent Itemset ======
fig1, ax1 = plt.subplots(figsize=(9, 6))
x = np.arange(len(df))
width = 0.25

bars1 = ax1.bar(x - width, df['n_fi_basic'], width, label='Basic', color='#4C72B0', edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(x, df['n_fi_opt'], width, label='Optimized', color='#55A868', edgecolor='white', linewidth=0.5)
bars3 = ax1.bar(x + width, df['n_fi_spmf'], width, label='SPMF', color='#C44E52', edgecolor='white', linewidth=0.5)

ax1.set_xlabel('Tập dữ liệu')
ax1.set_ylabel('Số lượng Frequent Itemset')
ax1.set_title('(a-1) So sánh số lượng Frequent Itemset', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df['dataset'], rotation=30, ha='right')
ax1.legend(fontsize=10)
ax1.set_yscale('log')
ax1.set_ylim(top=ax1.get_ylim()[1] * 5)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height):,}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=7, rotation=45)

fig1.tight_layout()
fig1.savefig(os.path.join(os.path.dirname(__file__), "a1_fi_count_comparison.png"), dpi=200, bbox_inches='tight')
print("Saved: a1_fi_count_comparison.png")

# ====== Hình 2: Tỉ lệ khớp (Match %) ======
fig2, ax2 = plt.subplots(figsize=(9, 6))
colors_match = ['#2ecc71' if m == 100.0 else '#e74c3c' for m in df['match_pct']]
bars_match = ax2.bar(df['dataset'], df['match_pct'], color=colors_match, edgecolor='white', linewidth=0.5, width=0.6)
ax2.set_xlabel('Tập dữ liệu')
ax2.set_ylabel('Tỉ lệ khớp (%)')
ax2.set_title('(a-2) Tỉ lệ itemset khớp hoàn toàn', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 115)
ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% target')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars_match, df['match_pct']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11,
             color='#2ecc71' if val == 100.0 else '#e74c3c')

ax2.set_xticklabels(df['dataset'], rotation=30, ha='right')
fig2.tight_layout()
fig2.savefig(os.path.join(os.path.dirname(__file__), "a2_match_pct.png"), dpi=200, bbox_inches='tight')
print("Saved: a2_match_pct.png")

# ====== Hình 3: So sánh thời gian chạy ======
fig3, ax3 = plt.subplots(figsize=(9, 6))
bars_ours = ax3.bar(x - 0.2, df['time_ours_ms'], 0.35, label='Our Implementation', color='#4C72B0', edgecolor='white')
bars_spmf = ax3.bar(x + 0.2, df['time_spmf_ms'], 0.35, label='SPMF', color='#C44E52', edgecolor='white')

ax3.set_xlabel('Tập dữ liệu')
ax3.set_ylabel('Thời gian chạy (ms)')
ax3.set_title('(a-3) So sánh thời gian chạy tại minsup kiểm tra', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(df['dataset'], rotation=30, ha='right')
ax3.legend(fontsize=10)
ax3.set_yscale('log')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

fig3.tight_layout()
fig3.savefig(os.path.join(os.path.dirname(__file__), "a3_runtime_comparison.png"), dpi=200, bbox_inches='tight')
print("Saved: a3_runtime_comparison.png")

plt.show()

# --- Print summary table ---
print("\n" + "="*80)
print("BẢNG TÓM TẮT TÍNH ĐÚNG ĐẮN")
print("="*80)
for _, row in df.iterrows():
    status = "✓ PASS" if row['match_pct'] == 100.0 else "✗ FAIL"
    print(f"  {row['dataset']:15s} | minsup={row['minsup_pct']:5.1f}% | "
          f"Basic={int(row['n_fi_basic']):>7,} | Opt={int(row['n_fi_opt']):>7,} | "
          f"SPMF={int(row['n_fi_spmf']):>7,} | Match={row['match_pct']:.1f}% | {status}")
print("="*80)
