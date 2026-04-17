"""
对比修改后的结果与答案
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from HydroArray.io import read_hydro_table, extract_time_series
from HydroArray.domain.process.runoff import saturation_excess_runoff


def main():
    print("="*80)
    print("对比修改后的结果与答案")
    print("="*80)

    data_file = Path(__file__).parent / "data.xlsx"
    answers_path = Path(__file__).parent / "answers.xlsx"

    # 读取数据
    raw_df = read_hydro_table(data_file)
    print(f"成功读取数据，共 {len(raw_df)} 条记录")

    pe_df = extract_time_series(raw_df, value_col=['P', 'Ep'], drop_na=False)
    print(f"提取完成，共 {len(pe_df)} 条记录")

    # 运行计算
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

    # 读取答案
    answers_df = pd.read_excel(answers_path)
    answers_df = answers_df[['月份', '日期', 'P', 'Ep', 'EU', 'EL', 'ED', 'E', 'PE', 'WU', 'WL', 'WD', 'W', 'R']]
    answers_df = answers_df.dropna(subset=['日期'])

    # 对比结果
    print("\n" + "="*80)
    print("详细对比结果")
    print("="*80)

    cols_to_compare = ['P', 'Ep', 'EU', 'EL', 'ED', 'E', 'PE', 'WU', 'WL', 'WD', 'W', 'R']

    diff_count = 0
    for i in range(min(30, len(result_df))):
        calc = result_df.iloc[i]
        ans = answers_df.iloc[i]

        day_diff = []
        for col in cols_to_compare:
            calc_val = calc[col]
            ans_val = ans[col]

            if pd.isna(calc_val) and pd.isna(ans_val):
                continue
            if pd.isna(calc_val) or pd.isna(ans_val):
                day_diff.append(f'{col}: NaN差异')
                continue

            diff = abs(calc_val - ans_val)
            if diff > 0.01:
                day_diff.append(f'{col}: 计算={calc_val:.4f}, 答案={ans_val:.4f}, 差={diff:.4f}')

        if day_diff:
            diff_count += 1
            print(f'\n第{i+1}天 (日期={calc.date}):')
            for d in day_diff:
                print(f'  {d}')

    if diff_count == 0:
        print('\n恭喜！所有天的计算结果都与答案匹配！')
    else:
        print(f'\n共有 {diff_count} 天存在差异')

    # 统计总体差异
    print('\n' + "="*80)
    print('总体统计')
    print("="*80)

    for col in ['EU', 'EL', 'ED', 'E', 'PE', 'WU', 'WL', 'WD', 'W', 'R']:
        diffs = []
        for i in range(len(result_df)):
            calc_val = result_df.iloc[i][col]
            ans_val = answers_df.iloc[i][col]
            if not pd.isna(calc_val) and not pd.isna(ans_val):
                diffs.append(abs(calc_val - ans_val))

        if diffs:
            max_diff = max(diffs)
            avg_diff = sum(diffs) / len(diffs)
            match_count = sum(1 for d in diffs if d < 0.01)
            print(f'{col}: 最大差异={max_diff:.4f}, 平均差异={avg_diff:.4f}, 匹配天数={match_count}/{len(diffs)}')


if __name__ == '__main__':
    main()
