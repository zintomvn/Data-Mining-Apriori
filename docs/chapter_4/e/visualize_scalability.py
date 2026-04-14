"""
Thực nghiệm (e): Khả năng mở rộng (Scalability)
Tách ra 3 hình riêng:
  1. Retail (sparse) — runtime vs DB size + linear/quadratic fit
  2. Accidents (dense) — runtime vs DB size + linear/quadratic fit
  3. So sánh chuẩn hóa cả 2 dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

script_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(script_dir, "scalability.csv"))
datasets = df['dataset'].unique()

colors = {'Retail': '#DD8452', 'Accidents': '#C44E52'}
markers = {'Retail': 'o', 'Accidents': 's'}

# ====== Hình 1 & 2: Từng dataset riêng ======
for idx, dataset in enumerate(datasets):
    fig, ax = plt.subplots(figsize=(9, 6))
    subset = df[df['dataset'] == dataset].sort_values('pct')

    pct_vals = subset['pct'].values
    n_trans = subset['n_trans'].values
    time_vals = subset['time_ms'].values

    color = colors.get(dataset, '#4C72B0')
    marker = markers.get(dataset, 'o')
    ds_type = 'Sparse' if dataset == 'Retail' else 'Dense'

    # Actual data
    ax.plot(pct_vals, time_vals, f'{marker}-', color=color, linewidth=2.5, markersize=8,
            label='Thực tế', markeredgecolor='white', markeredgewidth=1, zorder=3)

    # Linear fit
    coeffs = np.polyfit(pct_vals, time_vals, 1)
    linear_fit = np.polyval(coeffs, pct_vals)
    ax.plot(pct_vals, linear_fit, '--', color='gray', linewidth=1.5,
            label='Tuyến tính (fit)', alpha=0.7)

    # Quadratic fit
    coeffs2 = np.polyfit(pct_vals, time_vals, 2)
    x_smooth = np.linspace(pct_vals.min(), pct_vals.max(), 50)
    quad_fit = np.polyval(coeffs2, x_smooth)
    ax.plot(x_smooth, quad_fit, ':', color='#8172B3', linewidth=1.5,
            label='Bậc 2 (fit)', alpha=0.7)

    # R² for linear fit
    ss_res = np.sum((time_vals - linear_fit) ** 2)
    ss_tot = np.sum((time_vals - np.mean(time_vals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    ax.set_title(f'(e-{idx+1}) Scalability — {dataset} ({ds_type})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Kích thước CSDL (%)')
    ax.set_ylabel('Thời gian chạy (ms)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Secondary x-axis with transaction count
    ax2_top = ax.twiny()
    ax2_top.set_xlim(ax.get_xlim())
    ax2_top.set_xticks(pct_vals)
    ax2_top.set_xticklabels([f'{n:,}' for n in n_trans], fontsize=9)
    ax2_top.set_xlabel('Số giao dịch', fontsize=10)

    ax.annotate(f'R² (linear) = {r2:.4f}', xy=(0.05, 0.90), xycoords='axes fraction',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange', alpha=0.8))

    fig.tight_layout()
    fname = f"e{idx+1}_scalability_{dataset.lower()}.png"
    fig.savefig(os.path.join(script_dir, fname), dpi=200, bbox_inches='tight')
    print(f"Saved: {fname}")

# ====== Hình 3: So sánh chuẩn hóa ======
fig3, ax3 = plt.subplots(figsize=(9, 6))
for dataset in datasets:
    subset = df[df['dataset'] == dataset].sort_values('pct')
    pct_vals = subset['pct'].values
    time_vals = subset['time_ms'].values

    time_norm = (time_vals - time_vals.min()) / (time_vals.max() - time_vals.min() + 1e-9)

    color = colors.get(dataset, '#4C72B0')
    marker = markers.get(dataset, 'o')
    ds_type = 'Sparse' if dataset == 'Retail' else 'Dense'

    ax3.plot(pct_vals, time_norm, f'{marker}-', color=color, linewidth=2.5, markersize=8,
             label=f'{dataset} ({ds_type})', markeredgecolor='white', markeredgewidth=1)

ax3.plot([10, 100], [0, 1], '--', color='gray', linewidth=1.5, alpha=0.5, label='Tuyến tính lý tưởng')
ax3.set_title('(e-3) So sánh xu hướng scalability (chuẩn hóa)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Kích thước CSDL (%)')
ax3.set_ylabel('Thời gian (chuẩn hóa 0-1)')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')

fig3.tight_layout()
fig3.savefig(os.path.join(script_dir, "e3_scalability_normalized.png"), dpi=200, bbox_inches='tight')
print("Saved: e3_scalability_normalized.png")

plt.show()

# --- Summary ---
print("\n" + "="*80)
print("NHAN XET VE KHA NANG MO RONG")
print("="*80)
for dataset in datasets:
    subset = df[df['dataset'] == dataset].sort_values('pct')
    pct = subset['pct'].values
    time_vals = subset['time_ms'].values

    coeffs = np.polyfit(pct, time_vals, 1)
    linear_fit = np.polyval(coeffs, pct)
    ss_res = np.sum((time_vals - linear_fit) ** 2)
    ss_tot = np.sum((time_vals - np.mean(time_vals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    speedup = time_vals[-1] / time_vals[0]
    print(f"  {dataset:15s} | Tang {pct[0]:.0f}%->{pct[-1]:.0f}%: "
          f"thoi gian {time_vals[0]:,.1f} -> {time_vals[-1]:,.1f} ms (x{speedup:.1f}) | "
          f"R2(linear)={r2:.4f}")
print("="*80)
