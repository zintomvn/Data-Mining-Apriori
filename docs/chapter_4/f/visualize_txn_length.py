"""
Thực nghiệm (f): Ảnh hưởng của độ dài giao dịch trung bình
Tách ra 3 hình riêng:
  1. Thời gian chạy vs avg_len
  2. Số lượng FI vs avg_len
  3. Dual-axis: cả time lẫn FI
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

script_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(script_dir, "txn_length.csv"))

color_time = '#4C72B0'
color_fi = '#C44E52'

# ====== Hình 1: Thời gian chạy theo avg_len ======
fig1, ax1 = plt.subplots(figsize=(9, 6))
ax1.plot(df['avg_len'], df['time_ms'], 'o-', color=color_time, linewidth=2.5, markersize=8,
         markeredgecolor='white', markeredgewidth=1)
ax1.fill_between(df['avg_len'], 0, df['time_ms'], alpha=0.15, color=color_time)
ax1.set_xlabel('Độ dài giao dịch trung bình (avg_len)')
ax1.set_ylabel('Thời gian chạy (ms)')
ax1.set_title('(f-1) Thời gian chạy vs Độ dài giao dịch', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')

for _, row in df.iterrows():
    ax1.annotate(f"{row['time_ms']:.1f} ms",
                 xy=(row['avg_len'], row['time_ms']),
                 xytext=(5, 10), textcoords="offset points",
                 fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

fig1.tight_layout()
fig1.savefig(os.path.join(script_dir, "f1_runtime_vs_avglen.png"), dpi=200, bbox_inches='tight')
print("Saved: f1_runtime_vs_avglen.png")

# ====== Hình 2: Số lượng FI theo avg_len ======
fig2, ax2 = plt.subplots(figsize=(9, 6))
ax2.plot(df['avg_len'], df['n_fi'], 's-', color=color_fi, linewidth=2.5, markersize=8,
         markeredgecolor='white', markeredgewidth=1)
ax2.fill_between(df['avg_len'], 0, df['n_fi'], alpha=0.15, color=color_fi)
ax2.set_xlabel('Độ dài giao dịch trung bình (avg_len)')
ax2.set_ylabel('Số lượng Frequent Itemset')
ax2.set_title('(f-2) Số FI vs Độ dài giao dịch', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

for _, row in df.iterrows():
    ax2.annotate(f"{int(row['n_fi'])}",
                 xy=(row['avg_len'], row['n_fi']),
                 xytext=(5, 10), textcoords="offset points",
                 fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, "f2_fi_vs_avglen.png"), dpi=200, bbox_inches='tight')
print("Saved: f2_fi_vs_avglen.png")

# ====== Hình 3: Dual-axis (Time + FI count) ======
fig3, ax3 = plt.subplots(figsize=(9, 6))

line1 = ax3.plot(df['avg_len'], df['time_ms'], 'o-', color=color_time, linewidth=2.5,
                 markersize=8, markeredgecolor='white', markeredgewidth=1, label='Thời gian (ms)')
ax3.set_xlabel('Độ dài giao dịch trung bình (avg_len)')
ax3.set_ylabel('Thời gian chạy (ms)', color=color_time)
ax3.tick_params(axis='y', labelcolor=color_time)

ax3b = ax3.twinx()
line2 = ax3b.plot(df['avg_len'], df['n_fi'], 's--', color=color_fi, linewidth=2,
                  markersize=8, markeredgecolor='white', markeredgewidth=1, label='Số FI')
ax3b.set_ylabel('Số lượng Frequent Itemset', color=color_fi)
ax3b.tick_params(axis='y', labelcolor=color_fi)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper left', fontsize=10)
ax3.set_title('(f-3) Tổng hợp: Time & FI vs Avg Transaction Length', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')

fig3.tight_layout()
fig3.savefig(os.path.join(script_dir, "f3_combined_avglen.png"), dpi=200, bbox_inches='tight')
print("Saved: f3_combined_avglen.png")

plt.show()

# --- Summary ---
print("\n" + "="*80)
print("PHAN TICH ANH HUONG CUA DO DAI GIAO DICH")
print("="*80)
for _, row in df.iterrows():
    print(f"  avg_len={row['avg_len']:5.1f} | time={row['time_ms']:>8.2f} ms | n_fi={int(row['n_fi']):>5}")
print("-"*80)
if len(df) > 1:
    time_ratio = df['time_ms'].iloc[-1] / max(df['time_ms'].iloc[0], 0.01)
    print(f"\n  Khi avg_len tang tu {df['avg_len'].iloc[0]} -> {df['avg_len'].iloc[-1]}:")
    print(f"    * Thoi gian tang x{time_ratio:,.0f}")
    print(f"    * So FI tang tu {int(df['n_fi'].iloc[1])} -> {int(df['n_fi'].iloc[-1])}")
    print(f"\n  -> Thuat toan Apriori bi anh huong RAT MANH boi do dai giao dich")
    print(f"  -> Giao dich dai sinh ra nhieu candidate hon (C(n,k) tang theo n)")
print("="*80)
