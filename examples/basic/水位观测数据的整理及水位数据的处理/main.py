'''
瞬时水位和日均水位的转换示范
'''

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from HydroArray.io import read_hydro_table,extract_time_series

if __name__ == "__main__":
    print("="*60)
    print("HydroArray 水位观测数据的整理及水位数据的处理示例")
    print("="*60)

    data_file_1 = Path(__file__).parent / "1-1.xlsx"
    data_file_2 = Path(__file__).parent / "1-2.xlsx"
    data_file_3 = Path(__file__).parent / "1-3.xlsx"

    try:
        raw_df_1 = read_hydro_table(data_file_1)
        raw_df_2 = read_hydro_table(data_file_2)
        raw_df_3 = read_hydro_table(data_file_3)

        print(f"  数据列: {raw_df_1.columns.tolist()}")
        print(f"  时间范围: {raw_df_1['date'].min()} 至 {raw_df_1['date'].max()}")
        print("\n  原始数据预览:")
        print(raw_df_1.head(10).to_string())

        print(f"  数据列: {raw_df_2.columns.tolist()}")
        print(f"  时间范围: {raw_df_2['date'].min()} 至 {raw_df_2['date'].max()}")
        print("\n  原始数据预览:")
        print(raw_df_2.head(10).to_string())

        print(f"  数据列: {raw_df_3.columns.tolist()}")
        print(f"  时间范围: {raw_df_3['date'].min()} 至 {raw_df_3['date'].max()}")
        print("\n  原始数据预览:")
        print(raw_df_3.head(10).to_string())

    except Exception as e:
        print(f"✗ 读取数据失败: {e}")
        sys.exit(1)