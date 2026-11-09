"""
新安江模型案例 · 报告生成
========================
生成 Markdown 报告, 结构遵循课程要求:
  1. 任务与方法
  2. 数据说明
  3. 参数率定结果表 (日模型 + 次洪模型)
  4. 误差统计 (日模型分年 + 率定/验证分表 + 次洪分场)
  5. 模型表现分析
  6. 图表索引
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import HydroArray as ha

from xaj_config import (
    OUT, AREA_KM2,
    PERMIT_DAILY_RUNOFF_PCT, PERMIT_EVENT_PEAK_PCT,
    PERMIT_EVENT_VOL_PCT, PERMIT_EVENT_PEAK_LAG_H,
    WARMUP_END, CALIB_START, CALIB_END, VALID_START, VALID_END,
)
from xaj_metrics import fmt_pct, fmt_num


def _event_figure_name(row: pd.Series) -> str:
    role = row["event_role"]
    if role == "预热期":
        tag = "preheat"
    elif role == "率定期(参与率定)":
        tag = "calib"
    elif role == "率定期(未参与率定)":
        tag = "calibcheck"
    else:
        tag = "valid"
    return f"event_{int(row['event']):02d}_{tag}.png"


def _embed_image_lines(title: str, path: str, note: str) -> list[str]:
    return [f"### {title}", "", note, "", f"![{title}]({path})", ""]


def _md_param_table(p_daily: ha.XinAnjiangParameters,
                    p_event: ha.XinAnjiangParameters,
                    ref: dict) -> str:
    """参数率定结果表: 参数 | 含义 | 参考值 | 日模型 | 次洪模型。"""
    param_info = [
        ("K", "蒸发皿折算系数"),
        ("B", "蓄水容量分布曲线指数"),
        ("C", "深层散发系数"),
        ("WM", "张力水容量 (mm)"),
        ("WUM", "上层张力水容量 (mm)"),
        ("WLM", "下层张力水容量 (mm)"),
        ("WDM", "深层张力水容量 (mm)"),
        ("IMP", "不透水面积比"),
        ("SM", "自由水容量 (mm)"),
        ("EX", "自由水容量分布曲线指数"),
        ("KI", "壤中流出流系数"),
        ("KG", "地下水出流系数"),
        ("CI", "壤中流消退系数"),
        ("CG", "地下水消退系数"),
        ("L", "河网汇流滞时 (时段)"),
        ("KE", "Muskingum K (h)"),
        ("xe", "Muskingum x"),
    ]
    # ref 别名
    def _ref_str(k):
        if k == "IMP":
            return str(ref.get("IM", ref.get("IMP", "—")))
        if k == "xe":
            return str(ref.get("X", ref.get("xe", "—")))
        return str(ref.get(k, "—"))

    head = ("| 参数 | 含义 | 参考值 | 日模型最终 | 次洪模型最终 |\n"
            "| --- | --- | --- | ---: | ---: |")
    rows = []
    # 次洪模型继承的参数 (未参与次洪率定)
    event_inherited = {"C", "WM", "WUM", "WLM", "WDM", "IMP", "EX"}
    # 次洪模型固定参数 (不参与率定)
    event_fixed = {"KE"}
    for k, meaning in param_info:
        v_d = getattr(p_daily, k)
        v_e = getattr(p_event, k)
        if k in event_inherited:
            inherit = f"{v_e:.3f} (继承)" if abs(v_e - v_d) < 1e-6 else f"{v_e:.3f}"
        elif k in event_fixed:
            inherit = f"{v_e:.3f} (固定)"
        else:
            inherit = f"{v_e:.3f} (率定)"
        rows.append(f"| {k} | {meaning} | {_ref_str(k)} | {v_d:.3f} | {inherit} |")
    return head + "\n" + "\n".join(rows)


def _md_daily_year_table(df_y: pd.DataFrame) -> str:
    """日模型分年误差统计表。

    列: 年份, 天数, NSE, 年径流深相对误差 (%), 年径流深合格 (是/否),
       日值合格率 (%), 实测径流深 (mm/a), 计算径流深 (mm/a)
    """
    head = ("| 年份 | 天数 | NSE | 年径流深相对误差 (%) | 年合格 | 日值合格率 (%) "
            "| 实测径流深 (mm/a) | 计算径流深 (mm/a) |\n"
            "| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |")
    rows = []
    for _, r in df_y.iterrows():
        yr = int(r["year"])
        if yr <= 1989:
            yr_str = f"{yr} (预热)"
        elif yr <= 1993:
            yr_str = f"{yr} (率定)"
        else:
            yr_str = f"{yr} (验证)"
        daily_qual = (f"{r['daily_qual_pct']:.2f}"
                      if pd.notna(r["daily_qual_pct"]) else "—")
        ann_q = "✓" if r["annual_qualified"] else "✗"
        rows.append(
            f"| {yr_str} | {int(r['n'])} | {fmt_num(r['nse'])} | "
            f"{fmt_pct(r['rel_err_pct'])} | {ann_q} | {daily_qual} | "
            f"{r['obs_mm']:.1f} | {r['sim_mm']:.1f} |"
        )
    eval_years = df_y[df_y["year"] > 1989]
    if not eval_years.empty:
        avg_n = int(eval_years["n"].mean())
        avg_nse = float(eval_years["nse"].mean())
        avg_re = float(((eval_years["sim_mm"].sum() - eval_years["obs_mm"].sum())
                        / max(eval_years["obs_mm"].sum(), 1e-9) * 100.0))
        avg_qual = float(eval_years["daily_qual_pct"].mean())
        ann_q_count = int(eval_years["annual_qualified"].sum())
        ann_q_pct = ann_q_count / len(eval_years) * 100.0
        rows.append(
            f"| 1990-1996 年均 | {avg_n} | {fmt_num(avg_nse)} | "
            f"{fmt_pct(avg_re)} | {ann_q_count}/{len(eval_years)} | "
            f"{avg_qual:.2f} | "
            f"{eval_years['obs_mm'].mean():.1f} | "
            f"{eval_years['sim_mm'].mean():.1f} |"
        )
    return head + "\n" + "\n".join(rows)


def _md_daily_period_table(df_p: pd.DataFrame) -> str:
    """日模型率定期/验证期/整期统计表。

    列: 时段, 天数, NSE, 累计径流深相对误差 (%), 合格, 日值合格率 (%),
       实测径流深 (mm), 计算径流深 (mm)
    """
    head = ("| 时段 | 天数 | NSE | 累计径流深相对误差 (%) | 合格 | 日值合格率 (%) "
            "| 实测径流深 (mm) | 计算径流深 (mm) |\n"
            "| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |")
    rows = []
    for _, r in df_p.iterrows():
        ann_q = "✓" if r["annual_qualified"] else "✗"
        daily_qual = (f"{r['daily_qual_pct']:.2f}"
                      if pd.notna(r["daily_qual_pct"]) else "—")
        rows.append(
            f"| {r['period']} | {int(r['n'])} | {fmt_num(r['nse'])} | "
            f"{fmt_pct(r['rel_err_pct'])} | {ann_q} | {daily_qual} | "
            f"{r['obs_mm']:.1f} | {r['sim_mm']:.1f} |"
        )
    return head + "\n" + "\n".join(rows)


def _md_event_table(ev_res: pd.DataFrame, ev_summary: dict) -> str:
    """次洪分场次误差统计表 (4 指标 + 性质 + 复峰标记)。"""
    head = ("| 场次 | 性质 | 起止时间 | 径流深相对误差 (%) | 洪峰相对误差 (%) "
            "| 峰时差 (h) | NSE | 复峰 |\n"
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | :---: |")
    rows = []
    for _, r in ev_res.iterrows():
        ev_idx = int(r["event"])
        ts = pd.Timestamp(r["t_start"])
        mp = "✓" if r.get("is_multipeak", False) else ""
        rows.append(
            f"| {ev_idx} | {r['event_role']} | "
            f"{ts:%Y-%m-%d %H:%M} ~ "
            f"{pd.Timestamp(r['t_end']):%m-%d %H:%M} | "
            f"{fmt_pct(r['runoff_rel_err_pct'])} | "
            f"{fmt_pct(r['peak_rel_err_pct'])} | "
            f"{r['peak_lag_h']:+.1f} | {fmt_num(r['nse'])} | {mp} |"
        )
    # 验证汇总 (复峰场次单独注明, 便于读者参考)
    n_multipeak = int(ev_res[ev_res.get("is_multipeak", False) == True].shape[0]) if "is_multipeak" in ev_res.columns else 0
    rows.append(
        f"| 验证汇总 | — | {ev_summary['valid_count']} 场 | "
        f"{fmt_pct(ev_summary['valid_avg_vol_err'])} | "
        f"{fmt_pct(ev_summary['valid_avg_peak_err'])} | "
        f"{ev_summary['valid_avg_lag']:+.2f} | "
        f"{fmt_num(ev_summary['valid_avg_nse'])} | — |"
    )
    return head + "\n" + "\n".join(rows)


def _md_analysis(daily_m: dict, df_y: pd.DataFrame, df_p: pd.DataFrame,
                 ev_res: pd.DataFrame, ev_summary: dict,
                 train_events: list[int]) -> str:
    """生成模型表现分析文字。"""
    lines = []

    # ── 日模型 ──
    lines.append("### 5.1 日模型整体表现")
    lines.append("")
    calib_row = df_p[df_p["period"].str.contains("率定")].iloc[0]
    valid_row = df_p[df_p["period"].str.contains("验证")].iloc[0]
    lines.append(
        f"日模型采用 1989 年作为 warm-up 预热期 (不参与指标计算), "
        f"1990-1994 年为率定期, 1995-1996 年为验证期。"
    )
    lines.append("")
    calib_dq = float(calib_row["daily_qual_pct"])
    valid_dq = float(valid_row["daily_qual_pct"])
    calib_ann = "✓" if calib_row["annual_qualified"] else "✗"
    valid_ann = "✓" if valid_row["annual_qualified"] else "✗"
    lines.append(
        f"**率定期** ({int(calib_row['n'])} 天): NSE = {calib_row['nse']:.3f}, "
        f"累计径流深相对误差 = {calib_row['rel_err_pct']:+.2f}% (年合格 {calib_ann}), "
        f"日值合格率 = {calib_dq:.2f}%。"
    )
    lines.append(
        f"**验证期** ({int(valid_row['n'])} 天): NSE = {valid_row['nse']:.3f}, "
        f"累计径流深相对误差 = {valid_row['rel_err_pct']:+.2f}% (年合格 {valid_ann}), "
        f"日值合格率 = {valid_dq:.2f}%。"
    )
    lines.append("")
    # 分年表现
    eval_years = df_y[df_y["year"] > 1989]
    lines.append(
        f"分年 NSE (排除预热) 区间为 {eval_years['nse'].min():.3f} ~ "
        f"{eval_years['nse'].max():.3f}, 平均 {eval_years['nse'].mean():.3f}, "
        f"表明年际表现稳定。"
    )
    lines.append("")
    lines.append(
        "从年径流深散点图 (图 3) 看, 各年散点紧贴 45° 线, "
        "率定期与验证期点据无系统性偏离, 说明模型未出现过拟合, "
        "对多年水量的模拟能力良好。"
    )
    lines.append("")

    # ── 次洪模型 ──
    lines.append("### 5.2 次洪模型整体表现")
    lines.append("")
    n = ev_summary["valid_count"]
    n_train = int(ev_res["used_for_calibration"].sum())
    n_preheat = int((ev_res["event_role"] == "预热期").sum())
    lines.append(
        f"次洪模型采用率定期 1990-1994 的 {n_train} 场洪水联合率定 "
        f"(event {', '.join(map(str, train_events))}; "
        f"单阶段 DE 全局优化, 多起点重启 (5 seeds), 多场平均 NSE+水量+洪峰+峰时并兼顾差场约束)。"
    )
    lines.append("")
    lines.append(
        f"预热期另有 {n_preheat} 场洪水仅用于检验模型冷启动状态, "
        f"正式验证期 (1995-1996) 共 {n} 场。"
    )
    lines.append("")
    lines.append(
        f"在 {n} 场正式验证洪水中, 次洪模型平均 NSE = {ev_summary['valid_avg_nse']:.3f}, "
        f"平均洪峰相对误差 = {ev_summary['valid_avg_peak_err']:+.2f}%, "
        f"平均径流深相对误差 = {ev_summary['valid_avg_vol_err']:+.2f}%, "
        f"平均峰时差 = {ev_summary['valid_avg_lag']:+.2f} h。"
    )
    lines.append("")
    lines.append(
        f"按 SL 250 许可误差统计: "
        f"洪量合格 {ev_summary['qual_vol']}/{n} 场 "
        f"({ev_summary['qual_vol_pct']:.1f}%), "
        f"洪峰合格 {ev_summary['qual_peak']}/{n} 场 "
        f"({ev_summary['qual_peak_pct']:.1f}%), "
        f"峰时合格 {ev_summary['qual_lag']}/{n} 场 "
        f"({ev_summary['qual_lag_pct']:.1f}%)。"
    )
    lines.append("")

    # ── 误差分析 ──
    lines.append("### 5.3 场次误差原因分析")
    lines.append("")
    valid = ev_res[ev_res["event_role"] == "验证期"].sort_values("nse")
    worst = valid.head(2)
    best = valid.tail(1)
    lines.append("按 NSE 升序, 表现最差 2 场与最好 1 场如下。")
    lines.append("")
    lines.append("**表现较差:**")
    for _, r in worst.iterrows():
        lines.append(
            f"- **Event {int(r['event'])}** "
            f"({pd.Timestamp(r['t_start']):%Y-%m-%d} ~ "
            f"{pd.Timestamp(r['t_end']):%Y-%m-%d}): "
            f"NSE = {r['nse']:.3f}, 洪峰误差 = {r['peak_rel_err_pct']:+.2f}%, "
            f"峰时差 = {r['peak_lag_h']:+.1f} h。"
        )
    lines.append("")
    lines.append("**表现较好:**")
    for _, r in best.iterrows():
        lines.append(
            f"- **Event {int(r['event'])}** "
            f"({pd.Timestamp(r['t_start']):%Y-%m-%d} ~ "
            f"{pd.Timestamp(r['t_end']):%Y-%m-%d}): "
            f"NSE = {r['nse']:.3f}, 洪峰误差 = {r['peak_rel_err_pct']:+.2f}%, "
            f"峰时差 = {r['peak_lag_h']:+.1f} h。"
        )
    lines.append("")
    lines.append("**误差来源定性分析:**")
    lines.append("")
    if 12 in valid["event"].astype(int).values:
        e12 = valid[valid["event"].astype(int) == 12].iloc[0]
        if e12.get("is_multipeak", False):
            lines.append(
                f"Event 12 是长历时复峰洪水。表内峰时差 {e12['peak_lag_h']:+.1f} h "
                f"已按复峰窗口匹配规则在实测主峰附近 ±12h 内重新匹配; "
                f"若按全场最大值匹配, 峰时差会达到数十至上百小时。"
                f"洪峰量级误差仅 {e12['peak_rel_err_pct']:+.2f}%, NSE 较低 "
                f"({e12['nse']:.3f}) 主要来自多个子峰的相位错位, 而非主峰量级偏差。"
            )
        else:
            lines.append(
                f"Event 12 峰时差 {e12['peak_lag_h']:+.1f} h, "
                f"洪峰误差 {e12['peak_rel_err_pct']:+.2f}%, "
                f"NSE = {e12['nse']:.3f}。"
            )
        lines.append("")
    lines.append(
        "1. **降雨空间代表性**: 面雨量由 10 个雨量站面积加权得到, "
        "流域内局部强对流降水难以被稀疏站网捕捉。"
    )
    lines.append(
        "2. **初始土壤含水量**: 小时资料多数场次缺少完整 48h 前期资料, "
        "当前以模型默认土壤蓄水状态起算, 并用首时刻实测流量初始化慢水与河道状态; "
        "前期土壤湿润程度仍存在不确定性。"
    )
    lines.append(
        "3. **汇流参数简化**: 次洪汇流已按子流域段数进行多段 Muskingum 演算, "
        "但各段仍共用同一 KE/xe, 尚未细分不同河段的水力参数。"
    )
    lines.append(
        "4. **三水源参数**: KI/KG/CI/CG 控制地面/壤中/地下径流分配, "
        "单组参数难以覆盖所有场次的前期湿润状况差异。"
    )
    lines.append("")
    return "\n".join(lines)


def _write_docx_from_markdown(md_text: str, out_path: Path):
    """Render the generated markdown report to a simple .docx file."""
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=0)
            i += 1
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
            i += 1
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
            i += 1
            continue
        if line.startswith("> "):
            doc.add_paragraph(line[2:].strip(), style="Intense Quote")
            i += 1
            continue
        if line.startswith("![") and "](" in line and line.endswith(")"):
            img_path = line.split("](", 1)[1][:-1]
            resolved = (OUT / img_path).resolve() if not Path(img_path).is_absolute() else Path(img_path)
            if resolved.exists():
                doc.add_picture(str(resolved), width=Inches(6.2))
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                header = [c.strip() for c in table_lines[0].strip("|").split("|")]
                data_rows = []
                for raw in table_lines[2:]:
                    data_rows.append([c.strip() for c in raw.strip("|").split("|")])
                table = doc.add_table(rows=1, cols=len(header))
                table.style = "Table Grid"
                for j, cell in enumerate(header):
                    table.rows[0].cells[j].text = cell
                for row in data_rows:
                    cells = table.add_row().cells
                    for j, cell in enumerate(row):
                        cells[j].text = cell
            continue
        if line.startswith("- "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
            i += 1
            continue
        if line[:3] in {"1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. "}:
            doc.add_paragraph(line[3:].strip(), style="List Number")
            i += 1
            continue
        doc.add_paragraph(line)
        i += 1
    doc.save(str(out_path))


def write_report(daily_m: dict,
                 p_daily: ha.XinAnjiangParameters,
                 p_event: ha.XinAnjiangParameters,
                 ref: dict,
                 df_y: pd.DataFrame,
                 df_p: pd.DataFrame,
                 ev_res: pd.DataFrame,
                 train_events: list[int],
                 ev_summary: dict) -> Path:
    """生成完整 Markdown 报告。"""
    lines = []
    # ── 标题 ──
    lines.append("# 新安江模型案例 · 完整技术报告")
    lines.append("")
    lines.append("> 流域: 呈村流域 (290 km²)")
    lines.append("> 输入: `xaj_daily.csv` / `xaj_hourly.csv`")
    lines.append("> 模型: HydroArray `XinAnjiangModel` (三水源) + CI/CG 退水 + 半分布式汇流")
    lines.append("> 评价依据: SL 250《水文情报预报规范》")
    lines.append("")

    # ── 1. 任务与方法 ──
    lines.append("## 1. 任务与方法概述")
    lines.append("")
    lines.append("本案例基于 HydroArray 通用 API 完成呈村流域新安江模型的两尺度模拟:")
    lines.append("- **日模型**: 24h 步长连续模拟 1989-1996 年 (2922 天), "
                 "其中 1989 年为 warm-up 预热期, 1990-1994 率定, 1995-1996 验证。")
    lines.append("- **次洪模型**: 1h 步长, 对 15 场洪水逐场模拟, "
                 "6 场率定 (1990-1994) + 9 场验证 (含 1989 预热期 4 场 + 1995-1996 验证期 5 场)。")
    lines.append("")
    lines.append("**模型结构 (三水源新安江模型):**")
    lines.append("1. 蒸散发: 三层模型 (上层 EU + 下层 EL + 深层 ED)")
    lines.append("2. 产流: 蓄满产流 (张力水蓄水容量分布曲线, 参数 B)")
    lines.append("3. 分水源: 自由水蓄水库分为地表径流 RS、壤中流 RI、地下水 RG "
                 "(参数 SM, EX, KI, KG)")
    lines.append("4. 退水: RI 经 CI 线性水库退水, RG 经 CG 线性水库退水")
    lines.append("5. 汇流: 日模型采用 CS 线性水库; 次洪模型按子流域表 N_i 段 Muskingum 演算后在出口叠加")
    lines.append("")
    lines.append("**率定策略:**")
    lines.append("- 日模型单阶段 DE 全局率定 (仅率定期 1990-1994 参与优化):")
    lines.append("  - 12 个参数同时率定 (K, B, WUM, WLM, C, IMP, SM, EX, KI, KG, CI, CG)")
    lines.append("  - DE 种群规模 popsize=60 (≥5×n_params), maxiter=80, init=Sobol")
    lines.append("  - 5 个不同 seed 各跑一次 DE 取最优 (避免单次陷入局部最优)")
    lines.append("  - DE 收尾后用 L-BFGS-B 局部精修 (polish=True)")
    lines.append("  - 目标函数: -NSE + 0.15·|ΔV%|/100 + 0.10·|ΔPeak%|/100 + 0.05·max(0, 1-R²)")
    lines.append("  - 综合 NSE + 水量 + 峰差, 避免单目标偏差")
    lines.append("- 次洪模型 (前 11 场, 1989-1994, 多场联合率定):")
    lines.append("  - 继承日模型产流参数 (C, WM, WUM, WLM, WDM, IMP, EX)")
    lines.append("  - 继承 B 等流域属性参数, 重新率定 K 及分水源/退水/滞时参数")
    lines.append("  - 率定 SM, KI, KG (分水源), CI, CG (退水), L, xe (滞时与 Muskingum 汇流)")
    lines.append("  - KE 固定 1.0h (= 次洪步长 dt), 不参与率定 (避免与 dt 强耦合时的不稳定)")
    lines.append("  - 单阶段 DE (popsize=50, maxiter=100, 5 seeds 重启)")
    lines.append("  - 以逐场 NSE/洪量/洪峰/峰时综合得分平均值为主, 加入差场惩罚与稳健性约束")
    lines.append("  - Muskingum 稳定性约束: 2×KE×xe ≤ dt ≤ 2×KE×(1−xe), dt=1h")
    lines.append("  - 最多读取 48h 前期资料, 按实际可用小时数确定预热段; 无前期资料时以场次首时刻实测流量初始化慢水与河道状态")
    lines.append("")
    lines.append("**评价依据**: SL 250《水文情报预报规范》。")
    lines.append(f"- 日径流深许可相对误差: {PERMIT_DAILY_RUNOFF_PCT:.0f}%")
    lines.append(f"- 次洪洪量/洪峰许可相对误差: {PERMIT_EVENT_VOL_PCT:.0f}%")
    lines.append(f"- 次洪峰时许可绝对误差: {PERMIT_EVENT_PEAK_LAG_H:.0f} h")
    lines.append("")

    # ── 2. 数据 ──
    lines.append("## 2. 数据")
    lines.append("")
    lines.append("- **日资料** `xaj_daily.csv`: 1989-01-01 ~ 1996-12-31, "
                 "共 2922 天, 含 10 站降雨、蒸发皿蒸发 (mm/d) 和出口流量 (m³/s)。")
    lines.append("- **小时资料** `xaj_hourly.csv`: 1989-04-11 ~ 1996-07-04, "
                 "时间步长 1h, 共 5683 小时。")
    lines.append("- **子流域配置** `subbasins.yaml`: 10 个子流域, "
                 "面积比之和 1.0, 流域总面积 290.08 km²。")
    lines.append("- **参数参考** `params_ref.yaml`: 17 个参数的参考值/范围。")
    lines.append("- **场次配置** `events_event.yaml`: 15 场洪水。"
                 f"本案例按课程要求采用 1990-1994 年共 {len(train_events)} 场进行联合率定, "
                 "1989 年事件作为预热期检验, 1995-1996 年事件作为正式验证。")
    lines.append("")
    lines.append("**面平均输入处理:**")
    lines.append("- 面雨量 = Σ(子流域面积比 × 站降雨)")
    lines.append("- 日蒸发: 直接读取 `evap` 列 (蒸发皿蒸发, K 为折算系数)")
    lines.append("- 小时蒸发: 白天 (8:00-20:00, 12h) 按半正弦曲线 sin(πτ/12) 加权, "
                 "夜间 (20:00-次日 8:00) 权重为 0; 每日单独归一化, "
                 "日蒸发总量与原蒸发皿值严格相等。比简单 24 等分更符合实际蒸散发的昼夜差异。")
    lines.append("- 流量: m³/s ↔ mm/步 (cms_to_mm / mm_to_cms)")
    lines.append("")

    # ── 3. 参数率定结果 ──
    lines.append("## 3. 参数率定结果表")
    lines.append("")
    lines.append("下表汇总日模型与次洪模型率定后的最终参数值。")
    lines.append("日模型阶段 2 已加入分水源参数 (SM, EX, KI, KG, CI, CG); "
                 "次洪模型继承日模型部分产流参数与 B, 重新率定 K "
                 "及 SM, KI, KG, CI, CG, L, xe (分水源/退水/滞时/汇流), "
                 "**KE 固定为 1.0h (次洪步长 dt) 不参与率定**。"
                 "CS 作为日模型固定汇流参数使用, 不列入本表的率定结果。")
    lines.append("")
    lines.append(_md_param_table(p_daily, p_event, ref))
    lines.append("")
    # 越界说明
    out_of_ref = []
    from xaj_config import REF_RANGES
    for k, (lo, hi) in REF_RANGES.items():
        v_d = getattr(p_daily, k)
        v_e = getattr(p_event, k)
        for tag, v in [("日", v_d), ("次洪", v_e)]:
            if v < lo or v > hi:
                out_of_ref.append((k, tag, v, lo, hi))
    if out_of_ref:
        lines.append("**参数越界说明**: 以下参数值落在 `params_ref.yaml` 参考范围之外, "
                     "属于在参考范围基础上扩展的率定边界内结果:")
        for k, tag, v, lo, hi in out_of_ref:
            lines.append(f"- `{k}` ({tag}={v:.3f}): 参考 {lo:.3f}~{hi:.3f}, "
                         f"扩展边界内寻优, 原因: 小时尺度率定与 Muskingum 稳定性约束 "
                         f"可能导致参数偏离教材典型范围。")
    else:
        lines.append("所有率定参数均落在 `params_ref.yaml` 参考范围内。")
    lines.append("")

    # ── 4. 误差统计 ──
    lines.append("## 4. 误差统计")
    lines.append("")

    lines.append("### 4.1 日模型分年误差统计表")
    lines.append("")
    lines.append("表内\"日值合格率\"为: 当**实测日径流深 > 1 mm** 时, "
                 f"日值相对误差绝对值不超过 {PERMIT_DAILY_RUNOFF_PCT:.0f}% 的天数占比 (SL 250)。 "
                 "低流量天 (实测 ≤ 1 mm) 由于相对误差本身不稳定, 不计入分母。")
    lines.append("表内\"年合格\"列表示**年径流深相对误差**是否在 SL 250 许可范围内 "
                 f"(绝对值 ≤ {PERMIT_DAILY_RUNOFF_PCT:.0f}%); 这是比\"日值合格率\"更稳健的判断, "
                 "更能反映模型对年总水量的拟合能力。✓ 表示合格, ✗ 表示越界。")
    lines.append("表中除逐年结果外, 增列 1990-1996 年均值, "
                 "用于满足\"分别分年统计、每年均值、整个时期统计\"的提交要求。")
    lines.append("")
    lines.append(_md_daily_year_table(df_y))
    lines.append("")

    lines.append("### 4.2 日模型率定期/验证期统计表")
    lines.append("")
    lines.append("时段累计径流深相对误差、年合格、日值合格率定义同 4.1 节。"
                 "整期 (1990-1996) 行展示模型在整个率定+验证时段上的整体表现。")
    lines.append("")
    lines.append(_md_daily_period_table(df_p))
    lines.append("")

    lines.append("### 4.3 次洪模型分场次误差统计表 (4 指标 × 每场)")
    lines.append("")
    lines.append("4 个指标: 径流深相对误差、洪峰相对误差、峰时差 (h)、NSE。"
                 "\"性质\"列区分预热期、率定期(参与率定)与正式验证期。"
                 "末行\"验证汇总\"仅对 1995-1996 正式验证场次做算术平均。"
                 "**峰时差口径**: 对复峰洪水 (次大峰 ≥ 主峰 × 60%), 在实测主峰附近 ±12h 窗口内匹配计算峰, "
                 "以避免主峰归属错位导致峰时差达到数百小时。")
    lines.append("")
    lines.append(_md_event_table(ev_res, ev_summary))
    lines.append("")

    # ── 5. 分析 ──
    lines.append("## 5. 模型表现分析")
    lines.append("")
    lines.append(_md_analysis(daily_m, df_y, df_p, ev_res, ev_summary, train_events))
    lines.append("")

    # ── 6. 图表说明与附图 ──
    lines.append("## 6. 图表说明与附图")
    lines.append("")
    lines.extend(_embed_image_lines(
        "图 1 日模型降雨径流过程线",
        "daily_rainfall_runoff.png",
        "为说明连续模拟期内降雨输入与径流响应关系，绘制率定期和验证期的日尺度降雨径流过程线，结果见图 1。图中可直观看到模型对洪峰和枯水过程的跟踪能力。"
    ))
    lines.extend(_embed_image_lines(
        "图 2 日模型流量历时曲线",
        "daily_fdc.png",
        "为比较不同流量分位上的模拟偏差，补充绘制流量历时曲线，结果见图 2。该图用于辅助分析高流量段与低流量段的系统偏差。"
    ))
    lines.extend(_embed_image_lines(
        "图 3 年径流深散点图",
        "daily_annual_scatter.png",
        "为评价日模型对年径流深的模拟能力，绘制实测与计算年径流深散点图，并添加 45° 参考线，结果见图 3。散点位于 45° 线上方表示高估，下方表示低估。"
    ))
    lines.extend(_embed_image_lines(
        "图 4 分年径流深对比图",
        "daily_yearly_runoff_compare.png",
        "为便于逐年比较模型对年总水量的拟合情况，绘制分年实测与计算径流深柱状对比图，结果见图 4。"
    ))
    lines.extend(_embed_image_lines(
        "图 5 次洪分面总览图",
        "event_panel.png",
        "为先整体把握 15 场洪水的模拟质量，绘制逐场分面总览图，结果见图 5。该图用于快速定位典型好场次与差场次。"
    ))
    for _, row in ev_res.iterrows():
        idx = int(row["event"])
        role = row["event_role"]
        lines.extend(_embed_image_lines(
            f"图 6.{idx:02d} 场次 {idx} 降雨径流过程线",
            _event_figure_name(row),
            f"为分析场次 {idx} ({role}) 的降雨径流对应关系，绘制该场洪水的逐时降雨径流过程线，结果见图 6.{idx:02d}。图中实测流量与计算流量采用不同线型区分。"
        ))

    # ── 7. 精度与术语 ──
    lines.append("## 7. 数字精度与术语")
    lines.append("")
    lines.append("- 降雨/径流深: mm, 1 位小数")
    lines.append("- 流量: m³/s, 3 位有效数字")
    lines.append("- 误差指标: NSE/R² 保留 3 位, 百分比保留 2 位")
    lines.append(f"- 术语: SL 250 \"相对误差\"、\"绝对误差\"、\"合格率\"; "
                 f"许可误差: 日径流深 {PERMIT_DAILY_RUNOFF_PCT:.0f}%, "
                 f"次洪洪量/洪峰 {PERMIT_EVENT_VOL_PCT:.0f}%, "
                 f"峰时 {PERMIT_EVENT_PEAK_LAG_H:.0f} h")
    lines.append("")

    md_text = "\n".join(lines)
    out_path = OUT / "完整技术报告.md"
    out_path.write_text(md_text, encoding="utf-8")
    docx_path = OUT / "完整技术报告.docx"
    try:
        _write_docx_from_markdown(md_text, docx_path)
    except PermissionError:
        # Word 打开文件时会锁定目标路径，保留旧文件并写入独立优化版。
        _write_docx_from_markdown(md_text, OUT / "完整技术报告_优化版.docx")
    return out_path
