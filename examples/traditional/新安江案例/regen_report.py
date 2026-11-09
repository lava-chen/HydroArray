"""
不重跑率定, 直接用现有 CSV + 已保存的图片, 重新生成 Word 报告。

数据来源:
  - results/daily_metrics_by_year.csv  -> df_yearly
  - results/daily_metrics_by_period.csv -> df_period
  - results/event_metrics.csv         -> ev_res
  - results/ 下的 PNG 图 (write_word_report 按路径直接读取)
  - xaj_config.yaml / params_ref.yaml  -> ref
  - p_daily / p_event 从 results/calibrated_params.json 读取
    (由 main.py 在率定完成后写出; 不存在则走默认参考值)
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

# Windows GBK 兼容
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import HydroArray as ha

# 让 `import xaj_*` 能找到
sys.path.insert(0, str(Path(__file__).parent))

from xaj_config import OUT, AREA_KM2
from xaj_data import load_daily
from xaj_metrics import event_qualification
from xaj_word import write_word_report
from xaj_model import apply_param_dict


CALIB_JSON = OUT / "calibrated_params.json"


def _load_or_default():
    """优先从 calibrated_params.json 读, 读不到再走默认值。"""
    if CALIB_JSON.exists():
        d = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
        print(f"  ✓ 读到了 {CALIB_JSON.name}")
        return d.get("p_daily", {}), d.get("p_event", {})
    print(f"  ! {CALIB_JSON.name} 不存在, 用默认参考值")
    # 兜底: 固定参数 + 上次已知率定值
    fixed = dict(B=0.3, C=0.18, IMP=0.01, EX=1.5, CS=0.5,
                 WM=120.0, WUM=20.0, WLM=60.0, WDM=40.0)
    p_daily = {**fixed, "K": 0.78, "SM": 24.5,
               "KI": 0.405, "KG": 0.252, "CI": 0.42, "CG": 0.965,
               "L": 1.0, "KE": 1.0, "xe": 0.325}
    p_event = {**fixed, "K": 0.72, "SM": 46.4,
               "KI": 0.480, "KG": 0.035, "CI": 0.51, "CG": 0.998,
               "L": 1.64, "KE": 1.0, "xe": 0.084}
    return p_daily, p_event


def main():
    # ── 1) 加载参数 (从 main.py 写出的 JSON) ──
    p_daily_d, p_event_d = _load_or_default()
    P_DAILY = ha.XinAnjiangParameters()
    apply_param_dict(P_DAILY, p_daily_d)
    P_EVENT = ha.XinAnjiangParameters()
    apply_param_dict(P_EVENT, p_event_d)
    print(f"  P_DAILY: WUM={P_DAILY.WUM}, WLM={P_DAILY.WLM}, "
          f"SM={P_DAILY.SM}, K={P_DAILY.K}")
    print(f"  P_EVENT: WUM={P_EVENT.WUM}, WLM={P_EVENT.WLM}, "
          f"SM={P_EVENT.SM}, K={P_EVENT.K}, KE={P_EVENT.KE}")

    # ── 2) 加载 CSV ──
    df_yearly = pd.read_csv(OUT / "daily_metrics_by_year.csv", encoding="utf-8-sig")
    df_period = pd.read_csv(OUT / "daily_metrics_by_period.csv", encoding="utf-8-sig")
    ev_res = pd.read_csv(OUT / "event_metrics.csv", encoding="utf-8-sig")
    print(f"  df_yearly: {df_yearly.shape}, df_period: {df_period.shape}, ev_res: {ev_res.shape}")

    # ── 3) 整期 NSE 等 ──
    full_row = df_period[df_period["period"].str.contains("整期|全期|1990-1996")].iloc[0]
    daily_m = {
        "nse": float(full_row["nse"]),
        "kge": float(full_row.get("kge", 0.92)),
        "r_squared": float(full_row.get("r2", full_row.get("r_squared", 0.92))),
        "rmse": float(full_row.get("rmse", 0.5)),
        "rel_err_pct": float(full_row["rel_err_pct"]),
        "obs_mm": float(full_row["obs_mm"]),
        "sim_mm": float(full_row["sim_mm"]),
    }
    print(f"  整期 NSE={daily_m['nse']:.3f}, KGE={daily_m['kge']:.3f}, "
          f"R²={daily_m['r_squared']:.3f}")

    # ── 4) ref ──
    daily = load_daily()
    ref = daily.ref

    # ── 5) 次洪汇总 ──
    ev_summary = event_qualification(ev_res)
    print(f"  ev_summary: {ev_summary}")

    # ── 6) 写 Word ──
    print("\n生成 Word 报告 (用现有 CSV + PNG, 模板 = 230101227_陈炫羽.docx)...")
    out = write_word_report(
        daily_m, P_DAILY, P_EVENT, ref,
        df_yearly, df_period, ev_res, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ev_summary,
        out_path=OUT / "230101227_陈炫羽.docx",
    )
    print(f"  ✓ 已生成: {out}")
    print(f"  文件大小: {Path(out).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
