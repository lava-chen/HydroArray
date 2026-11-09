"""数据读写相关命令"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from .output import success_result


def register(subparsers: argparse.ArgumentParser) -> None:
    """注册 io 子命令

    数据读写操作模块，包括：
    - read-table: 读取水文观测表格
    - parse: 解析文件名获取元数据
    - batch: 批量解析文件夹
    - convert: 数据格式转换
    """
    io_parser = subparsers.add_parser(
        "io",
        help="数据读写操作",
        description="水文数据文件的读取、解析、格式转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的文件格式:
  读取: CSV, Excel (.xlsx, .xls), JSON
  转换: CSV, Excel, NetCDF, Zarr, ASC, GeoTIFF

示例:
  # 读取水文表格
  hydro io read-table --path data.csv --datetime timestamp

  # 批量解析文件名
  hydro io batch --folder /data --pattern "*.csv"

  # 格式转换
  hydro io convert -i data.csv -o data.nc --from-format csv --to-format netcdf
        """
    )

    sub = io_parser.add_subparsers(dest="io_action", required=True,
                                   help="操作类型")

    # 读取水文表格
    _register_read_table(sub)

    # 解析文件
    _register_parse(sub)

    # 批量解析
    _register_batch_parse(sub)

    # 转换格式
    _register_convert(sub)


def _register_read_table(sub) -> None:
    """注册读取表格子命令"""
    p = sub.add_parser(
        "read-table",
        help="读取水文观测表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="读取水文观测数据表格并输出为 JSON",
        epilog="""
输入文件格式:
  - CSV: 逗号分隔值文件
  - Excel: .xlsx 或 .xls 文件
  - JSON: JSON 数组格式

输出格式 (JSON):
  {
    "rows": 100,
    "columns": ["datetime", "precipitation", "flow"],
    "data": [
      {"datetime": "2024-01-01", "precipitation": 10.5, "flow": 25.3},
      ...
    ]
  }

提示: 使用 --compact 输出更紧凑的 JSON (无缩进)
        """
    )
    p.add_argument(
        "--path", "-p",
        required=True,
        help="数据文件路径"
    )
    p.add_argument(
        "--format", "-f",
        choices=["csv", "excel", "json", "auto"],
        default="auto",
        help="指定文件格式 (默认: auto 自动检测)"
    )
    p.add_argument(
        "--sheet", "-s",
        help="Excel 工作表名称或索引 (默认: 第一个工作表)"
    )
    p.add_argument(
        "--columns", "-c",
        help="要读取的列名 (逗号分隔，如: datetime,precipitation,flow)"
    )
    p.add_argument(
        "--datetime", "-dt",
        help="日期时间列名 (用于时间序列解析)"
    )
    p.add_argument(
        "--parse-dates",
        action="store_true",
        help="自动解析日期列"
    )
    p.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="跳过文件开头的行数 (默认: 0)"
    )
    p.add_argument(
        "--limit", "-n",
        type=int,
        help="限制读取的行数 (默认: 全部)"
    )
    p.add_argument(
        "--output", "-o",
        choices=["json", "text", "compact"],
        help="输出格式 (默认: 继承全局设置)"
    )
    p.set_defaults(func=_read_table)


def _register_parse(sub) -> None:
    """注册单文件解析子命令"""
    p = sub.add_parser(
        "parse",
        help="解析文件名获取元数据",
        description="从文件名中提取日期、类型等信息",
        epilog="""
支持的文件名模式:
  - 日期格式: YYYYMMDD, YYYY-MM-DD, YYYYMMDDHHmm
  - 站点编码: ST001, STA_001
  - 数据类型: PRECIP, FLOW, TEMP
  - 组合: ST001_FLOW_20240101.csv

示例:
  hydro io parse --path "data/ST001_PRECIP_20240101.csv"
  # 输出: {"station": "ST001", "type": "PRECIP", "date": "2024-01-01"}

  hydro io parse --path "/2024/01/data.csv"
  # 输出: {"folder_date": "2024/01", "filename": "data.csv"}
        """
    )
    p.add_argument(
        "--path", "-p",
        required=True,
        help="文件路径"
    )
    p.add_argument(
        "--type", "-t",
        choices=["filename", "folder", "path"],
        default="path",
        help="解析类型: filename=只解析文件名, folder=解析文件夹, path=完整路径"
    )
    p.add_argument(
        "--output", "-o",
        choices=["json", "text", "compact"],
        help="输出格式 (默认: 继承全局设置)"
    )
    p.set_defaults(func=_parse_file)


def _register_batch_parse(sub) -> None:
    """注册批量解析子命令"""
    p = sub.add_parser(
        "batch",
        help="批量解析文件夹中的文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="批量解析文件夹中所有匹配的文件",
        epilog="""
文件匹配模式 (glob):
  *.csv           - 所有 CSV 文件
  *_PRECIP*.csv   - 以 PRECIP 开头的文件
  2024*.nc        - 2024 年的 NetCDF 文件
  **/*.csv        - 递归搜索所有 CSV 文件

输出内容:
  - 每个文件的解析结果
  - 失败文件的错误信息
  - 汇总统计 (总数、成功数、失败数)
        """
    )
    p.add_argument(
        "--folder", "-f",
        required=True,
        help="文件夹路径"
    )
    p.add_argument(
        "--pattern", "-p",
        default="*.csv",
        help="文件匹配模式 (默认: *.csv)"
    )
    p.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="递归搜索子文件夹"
    )
    p.add_argument(
        "--max-files", "-n",
        type=int,
        help="限制处理的文件数量 (默认: 全部)"
    )
    p.add_argument(
        "--output", "-o",
        choices=["json", "text", "compact"],
        help="输出格式 (默认: 继承全局设置)"
    )
    p.set_defaults(func=_batch_parse)


def _register_convert(sub) -> None:
    """注册格式转换子命令"""
    p = sub.add_parser(
        "convert",
        help="转换数据格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="在多种水文数据格式之间转换",
        epilog="""
支持的格式:
  输入: csv, excel, netcdf, zarr
  输出: csv, excel, netcdf, zarr, asc, geotiff

格式说明:
  - CSV: 逗号分隔值，易于阅读和编辑
  - Excel: 二进制表格，保留格式
  - NetCDF: NetCDF 格式，适合科学数据
  - Zarr: 分块压缩格式，适合大数据
  - ASC: ArcGIS 栅格格式
  - GeoTIFF: GeoTIFF 栅格格式

注意:
  - ASC/GeoTIFF 只能用于栅格数据
  - Zarr 支持可选压缩 (--compression)
        """
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="输入文件路径"
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        help="输出文件路径"
    )
    p.add_argument(
        "--from-format", "-f",
        choices=["csv", "excel", "netcdf", "zarr", "auto"],
        default="auto",
        help="源格式 (默认: auto 自动检测)"
    )
    p.add_argument(
        "--to-format", "-t",
        required=True,
        choices=["csv", "excel", "netcdf", "zarr", "asc", "geotiff"],
        help="目标格式"
    )
    p.add_argument(
        "--compression", "-c",
        choices=["zlib", "lzf", "none"],
        default="zlib",
        help="NetCDF/Zarr 压缩方式 (默认: zlib)"
    )
    p.add_argument(
        "--keep-metadata",
        action="store_true",
        help="保留元数据信息"
    )
    p.set_defaults(func=_convert_format)


def _read_table(args) -> Dict:
    """读取水文表格"""
    from HydroArray.io.readers.table_reader import read_hydro_table
    import pandas as pd

    kwargs = {}
    if args.format != "auto":
        kwargs["format"] = args.format
    if args.sheet:
        kwargs["sheet"] = args.sheet
    if args.columns:
        kwargs["columns"] = [c.strip() for c in args.columns.split(',')]
    if args.datetime:
        kwargs["datetime_column"] = args.datetime
    if args.parse_dates:
        kwargs["parse_dates"] = True
    if args.skip_rows > 0:
        kwargs["skiprows"] = args.skip_rows

    df = read_hydro_table(args.path, **kwargs)

    if args.limit:
        df = df.head(args.limit)

    return success_result(
        data={
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        },
        meta={
            "file": args.path,
            "format": args.format if args.format != "auto" else "detected"
        }
    )


def _parse_file(args) -> Dict:
    """解析单个文件"""
    from HydroArray.utils.file_parser import parse_filename, parse_folder, parse_path

    if args.type == "filename":
        result = parse_filename(args.path)
    elif args.type == "folder":
        result = parse_folder(args.path)
    else:
        result = parse_path(args.path)

    return success_result(
        data=result,
        meta={"input_type": args.type}
    )


def _batch_parse(args) -> Dict:
    """批量解析文件"""
    from HydroArray.utils.file_parser import batch_parse_files
    from pathlib import Path

    folder = Path(args.folder)

    if args.recursive:
        files = list(folder.rglob(args.pattern))
    else:
        files = list(folder.glob(args.pattern))

    if args.max_files:
        files = files[:args.max_files]

    results = []
    success_count = 0
    fail_count = 0

    for f in files:
        try:
            parsed = parse_filename(f.name)
            results.append({
                "path": str(f),
                "parsed": parsed,
            })
            success_count += 1
        except Exception as e:
            results.append({
                "path": str(f),
                "error": str(e),
            })
            fail_count += 1

    return success_result(
        data={
            "results": results,
            "summary": {
                "total": len(files),
                "success": success_count,
                "failed": fail_count
            }
        },
        meta={"folder": str(folder), "pattern": args.pattern}
    )


def _convert_format(args) -> Dict:
    """转换数据格式"""
    import pandas as pd
    from pathlib import Path

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 读取数据
    if args.from_format == "auto":
        ext = input_path.suffix.lower()
        format_map = {".csv": "csv", ".xlsx": "excel", ".xls": "excel",
                      ".nc": "netcdf", ".netcdf": "netcdf", ".zarr": "zarr"}
        from_fmt = format_map.get(ext, "csv")
    else:
        from_fmt = args.from_format

    if from_fmt == "csv":
        df = pd.read_csv(input_path)
    elif from_fmt == "excel":
        df = pd.read_excel(input_path)
    elif from_fmt == "netcdf":
        import xarray as xr
        ds = xr.open_dataset(input_path)
        df = ds.to_dataframe()
    else:
        raise ValueError(f"不支持的输入格式: {from_fmt}")

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入数据
    if args.to_format == "csv":
        df.to_csv(output_path, index=False)
    elif args.to_format == "excel":
        df.to_excel(output_path, index=False)
    elif args.to_format == "netcdf":
        import xarray as xr
        compression = None if args.compression == "none" else args.compression
        df.to_xarray().to_netcdf(output_path, encoding={col: {"compress": compression}
                          for col in df.columns} if compression else {})
    else:
        raise ValueError(f"不支持的输出格式: {args.to_format}")

    return success_result(
        data={
            "input": str(input_path),
            "output": str(output_path),
            "rows": len(df),
            "columns": list(df.columns),
        },
        meta={
            "from_format": from_fmt,
            "to_format": args.to_format
        }
    )