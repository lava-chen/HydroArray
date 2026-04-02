"""
style_showcase.py — 风格展示脚本
运行: python style_showcase.py
输出: figures/style_showcase_{style}.png
"""

import sys, os
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

# 直接导入，绕过包的 __init__ 路径问题
import importlib.util
def load_style_module():
    spec = importlib.util.spec_from_file_location("styles", os.path.join(ROOT, "hydroarray", "plotting", "styles.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

sm = load_style_module()
use_hydro_style = sm.use_hydro_style
list_styles = sm.list_styles
get_colors = sm.get_colors

import numpy as np
import matplotlib.pyplot as plt

# ── 测试数据 ──────────────────────────────────────────────────────────────
np.random.seed(42)
t = np.arange(0, 365, 1)
obs  = 50 + 20 * np.sin(2 * np.pi * t / 365) + np.random.randn(365) * 8
sim1 = 48 + 22 * np.sin(2 * np.pi * t / 365 + 0.3) + np.random.randn(365) * 10
sim2 = 52 + 18 * np.sin(2 * np.pi * t / 365 - 0.2) + np.random.randn(365) * 12

out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)

styles = list(list_styles().keys())

# ── 图1: 时序图（双模型对比） ──────────────────────────────────────────
for style in styles:
    use_hydro_style(style, chinese=False)
    colors = get_colors(style)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, obs,  label="Observed",  color=colors[0], lw=1.5, alpha=0.8)
    ax.plot(t, sim1, label="Model A",   color=colors[1], lw=1.2, alpha=0.85)
    ax.plot(t, sim2, label="Model B",   color=colors[2], lw=1.2, alpha=0.85, linestyle="--")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Streamflow (m³/s)")
    ax.set_title(f"[{style.upper()}] Streamflow Simulation vs Observation", fontsize=12)
    ax.legend(framealpha=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, f"hydrograph_{style}.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ {style} → {path}")

# ── 图2: 散点图（NSE/KGE 评估） ─────────────────────────────────────
for style in styles[:4]:  # 只画前4个风格，省时间
    use_hydro_style(style, chinese=False)
    colors = get_colors(style)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(obs, sim1, s=12, alpha=0.6, color=colors[1], label="Model A")
    # 1:1线
    vmin, vmax = min(obs.min(), sim1.min()), max(obs.max(), sim1.max())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, label="1:1 Line", alpha=0.6)
    ax.set_xlabel("Observed (m³/s)")
    ax.set_ylabel("Simulated (m³/s)")
    ax.set_title(f"[{style.upper()}] Scatter Plot", fontsize=12)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"scatter_{style}.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ {style} scatter → {path}")

# ── 图3: 6风格并排对比（拼图） ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, style in enumerate(styles[:6]):
    use_hydro_style(style, chinese=False)
    colors = get_colors(style)
    ax = axes[i]
    ax.plot(t[:100], obs[:100],  color=colors[0], lw=1.5, label="Obs")
    ax.plot(t[:100], sim1[:100], color=colors[1], lw=1.2, label="Model")
    ax.set_title(style.upper(), fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
fig.suptitle("HydroArray Style Gallery — 6 Styles Comparison", fontsize=13)
fig.tight_layout()
path = os.path.join(out_dir, "gallery_comparison.png")
fig.savefig(path, dpi=150)
plt.close()
print(f"\n  ✓ Gallery → {path}")
print("\nAll done!")
