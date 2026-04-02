"""
大断面计算示例
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import HydroArray as ha

ha.use_hydro_style(style="dark_pro", dpi=100)

if __name__ == "__main__":
    print("="*60)
    print(f"HydroArray v{ha.__version__} - 大断面计算示例")
    print("="*60)

    data_file = Path(__file__).parent / "2-1.xlsx"

    try:
        raw_df = ha.read_hydro_table(data_file)
        print(f"✓ 成功读取数据，共 {len(raw_df)} 条记录")
        print(f"  数据列: {raw_df.columns.tolist()}")

        print("\n  数据预览:")
        print(raw_df.to_string())

        area_df, channel_section_result = ha.calculate_cross_section_area_detailed(
            raw_df, 
            given_elevation_list=[
                0.95, 5.80, 7.55, 8.35, 10.45, 10.85, 11.35, 11.45, 12.38, 12.42,
                13.58, 14.69, 15.11, 17.23, 17.38, 17.66, 18.21, 19.93, 20.21, 22.06
            ], 
            lowest_elevation=5.80
        )
        print("\n  计算结果:")
        formatted_df = area_df.copy()
        for col in ['L_dist(m)', 'R_dist(m)']:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{int(x)}" if x >= 100 else f"{x:.1f}"
            )
        print(formatted_df.to_string(index=False))
        print(channel_section_result.to_string(index=False))

        ha.cross_section_plot(raw_df, language="en")

    except Exception as e:
        print(f"✗ 读取数据失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
