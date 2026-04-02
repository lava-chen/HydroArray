"""
演示如何通过卫星数据画图
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import HydroArray as ha


if __name__ == "__main__":
    print("="*60)
    print(f"HydroArray v{ha.__version__} - 卫星数据画图示例")
    print("="*60)

    data_folder = Path(__file__).parent.parent / "data" / "fy3g" / "csv_h"

    print(f"数据目录: {data_folder}")

    df = ha.read_satellite_folder(data_folder)
    print(f"\n数据量: {len(df)} 条记录")

    times = ha.get_available_times(df)
    print(f"时间点数量: {len(times)}")
    print(f"时间范围: {times[0]} ~ {times[-1]}")

    print("\n" + "="*60)
    print("绘制卫星数据栅格图:")
    print("="*60)

    print("\n1. 绘制第一个时间点 (time=0):")
    ha.satellite_plot(
        df,
        time=0,
        cmap="Blues",
        style="hess"
    )

    print("\n2. 绘制最后一个时间点 (time=-1):")
    ha.satellite_plot(
        df,
        time=-1,
        cmap="precip",
        style="nature"
    )

    print("\n3. 带底图的栅格图:")
    ha.satellite_plot(
        df,
        time=0,
        show_basemap=True,
        cmap="precip",
        style="nature"
    )

    print("\n绘图完成！")
