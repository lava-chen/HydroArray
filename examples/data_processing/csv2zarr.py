"""
数据转换示范，将csv文件转换成zarr文件
Zarr 是一种专为大规模数值数组（多维数组，如气象、水文、基因组数据）设计的切片、压缩、分块存储格式。它在云计算和大数据分析场景中非常流行，常被视为 HDF5 的现代化替代方案。

优势：
- 云原生：支持 S3、GCS 等对象存储
- 并行读写：多进程/多线程安全
- 切片访问：只读取需要的部分数据
- 高效压缩：内置多种压缩算法

对于这个示例：
- csv格式的占用空间是510MB，而zarr只需要占用2.37MB
- 读取速度上csv需要花费6秒左右，而zarr只需要0.0368秒
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import HydroArray as ha


if __name__ == "__main__":
    print("="*60)
    print(f"HydroArray v{ha.__version__} - CSV 转 Zarr 示例")
    print("="*60)

    data_folder = Path(__file__).parent.parent / "data" / "fy3g" / "csv_h"
    output_folder = Path(__file__).parent.parent / "data" / "fy3g" / "zarr"

    print(f"\n输入目录: {data_folder}")
    print(f"输出目录: {output_folder}")

    print("\n" + "-"*60)
    print("步骤 1: 读取 CSV 并转换为 xarray Dataset")
    print("-"*60)

    start = time.time()
    ds = ha.read_satellite_folder(data_folder, output_format="dataset")
    csv_time = time.time() - start
    print(f"✓ 成功读取数据 (耗时: {csv_time:.2f} 秒)")
    print(ds)

    print("\n" + "-"*60)
    print("步骤 2: 保存为 Zarr 格式")
    print("-"*60)

    output_folder.mkdir(parents=True, exist_ok=True)
    zarr_path = output_folder / "fy3g_precip.zarr"

    start = time.time()
    ds.to_zarr(zarr_path, mode="w")
    save_time = time.time() - start
    print(f"✓ 已保存到: {zarr_path} (耗时: {save_time:.2f} 秒)")

    import os
    total_size = sum(
        os.path.getsize(zarr_path / f) 
        for f in os.listdir(zarr_path) 
        if os.path.isfile(zarr_path / f)
    )
    print(f"  文件大小: {total_size / 1024 / 1024:.2f} MB")

    print("\n" + "-"*60)
    print("步骤 3: 读取速度对比测试")
    print("-"*60)

    import xarray as xr

    start = time.time()
    ds_zarr = xr.open_zarr(zarr_path)
    _ = ds_zarr['value'].values
    zarr_time = time.time() - start
    print(f"Zarr 读取耗时: {zarr_time:.4f} 秒")

    print(f"\n速度对比:")
    print(f"  CSV 读取:  {csv_time:.2f} 秒")
    print(f"  Zarr 读取: {zarr_time:.4f} 秒")
    print(f"  加速比:    {csv_time / zarr_time:.1f}x")

    print("\n" + "-"*60)
    print("步骤 4: 切片访问测试")
    print("-"*60)

    start = time.time()
    single_time = ds_zarr['value'].isel(datetime=0).values
    slice_time = time.time() - start
    print(f"读取单个时间点: {slice_time:.4f} 秒")
    print(f"数据形状: {single_time.shape}")

    print("\n" + "="*60)
    print("转换完成！")
    print("="*60)

    print("\n使用方式:")
    print(f"  import xarray as xr")
    print(f"  ds = xr.open_zarr('{zarr_path}')")
    print(f"  ds['value'].isel(datetime=0).plot()")
