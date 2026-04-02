"""
读取器模块
"""

from HydroArray.io.readers.table_reader import (
    read_hydro_table,
    extract_time_series,
    detect_table_structure,
    TableType,
)

__all__ = [
    "read_hydro_table",
    "extract_time_series",
    "detect_table_structure",
    "TableType",
]
