"""
Input/Output Module for HydroArray

Provides readers for various data formats used in hydrological modeling.
"""

from HydroArray.io.readers.table_reader import (
    read_hydro_table,
    extract_time_series,
    detect_table_structure,
    TableType,
)

from HydroArray.io.grid_reader import (
    ASCGridReader,
    GeoTIFFReader,
    GridStackReader,
    GridMetadata,
    read_grid,
)

from HydroArray.io.forcings import (
    TimeSeriesReader,
    TimeSeriesMetadata,
    BasinForcingReader,
    read_forcing,
)

__all__ = [
    # Table reader
    "read_hydro_table",
    "extract_time_series",
    "detect_table_structure",
    "TableType",
    # Grid reader
    "ASCGridReader",
    "GeoTIFFReader",
    "GridStackReader",
    "GridMetadata",
    "read_grid",
    # Time series / forcing reader
    "TimeSeriesReader",
    "TimeSeriesMetadata",
    "BasinForcingReader",
    "read_forcing",
]
