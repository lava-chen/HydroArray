"""
流域蒸散发量及产流量计算示例

本示例演示如何使用 HydroArray 库进行水文计算。
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from HydroArray.io import read_hydro_table,extract_time_series
from HydroArray.domain.process.runoff import saturation_excess_runoff, two_source_runoff_separation, three_source_runoff_separation


if __name__ == "__main__":
    print("="*60)
    print("HydroArray 流域蒸散发量及产流量计算示例")
    print("="*60)

    data_file = Path(__file__).parent / "data.xlsx"
    result_file = Path(__file__).parent / "result.xlsx"

    try:
        raw_df = read_hydro_table(data_file)
        print(f"✓ 成功读取数据，共 {len(raw_df)} 条记录")
        print(f"  数据列: {raw_df.columns.tolist()}")
        print(f"  时间范围: {raw_df['date'].min()} 至 {raw_df['date'].max()}")

        print("\n  原始数据预览:")
        print(raw_df.head().to_string())

    except Exception as e:
        print(f"✗ 读取数据失败: {e}")
        sys.exit(1)

    pe_df = extract_time_series(raw_df, value_col=['P', 'Ep'], drop_na=False)
    print(f"✓ 提取完成，共 {len(pe_df)} 条记录")
    print("\n  P 和 Ep 数据预览:")
    print(pe_df.head(10).to_string())

    result_df = saturation_excess_runoff(
        pe_df,
        WUM=20.0,
        WLM=60.0,
        WDM=40.0,
        C=1/6,
        Kc=0.95,
        b=0.3,
        initial_WU=20.0,
        initial_WL=60.0,
        initial_WD=40.0,
        initial_R=0.0
    )
    print("\n蓄满产流计算完成:")
    print(result_df.head().to_string())

    # 二水源划分
    result_df_2 = two_source_runoff_separation(
        result_df,
        FC = 2.5
    )

    # 三水源划分
    result_df_3 = three_source_runoff_separation(
        result_df,
        SM=15.0,
        EX=1.5,
        KI=0.35,
        KG=0.35,
        initial_S=10.0
    )

    final_df = pd.merge(result_df, result_df_2, on='date', how='left')
    final_df = pd.merge(final_df, result_df_3, on='date', how='left')

    # 计算 R/PE 比值
    final_df['R_PE'] = final_df.apply(
        lambda row: row['R'] / row['PE'] if row['R'] != 0 else 0, axis=1
    )

    print("\n三水源划分计算完成:")
    print(final_df.head(15).to_string())

    # 保存结果
    final_df.to_excel(result_file, index=False)

    print("\n" + "="*60)
    print("校核计算")
    print("="*60)

    checks = [
        ("EU + EL + ED = E", (final_df['EU'] + final_df['EL'] + final_df['ED'] - final_df['E']).abs().max()),
        ("ΣPE = ΣP - ΣE", abs(final_df['PE'].sum() - (final_df['P'].sum() - final_df['E'].sum()))),
        ("ΣR = ΣPE - ΔW", abs(final_df['R'].sum() - (final_df['PE'].sum() - (final_df.iloc[-1]['W'] - final_df.iloc[0]['W'])))),
    ]
    for name, err in checks:
        print(f"{name}: {'通过 ✓' if err < 1e-6 else '未通过 ✗'} (误差: {err:.2e})")

    print("\n计算完成！")
