"""
Time Series Reader for Hydrological Model Forcing Data

Supports CSV, Excel, and NetCDF formats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class TimeSeriesMetadata:
    """Metadata for time series data."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timestep: str = "daily"  # daily, hourly, subhourly
    variables: List[str] = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = []


class TimeSeriesReader:
    """Reader for time series forcing data.

    Supports:
    - CSV files (with date column)
    - Excel files
    - NetCDF files
    """

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.metadata = TimeSeriesMetadata()
        self.data: Optional[pd.DataFrame] = None

    def read_csv(self, date_column: str = "date",
                 parse_dates: bool = True) -> pd.DataFrame:
        """Read time series from CSV file.

        Args:
            date_column: Name of the date column.
            parse_dates: Whether to parse dates.

        Returns:
            DataFrame with datetime index.
        """
        try:
            if parse_dates:
                self.data = pd.read_csv(self.filepath, parse_dates=[date_column])
                self.data = self.data.set_index(date_column)
            else:
                self.data = pd.read_csv(self.filepath)

            self.metadata.start_date = str(self.data.index[0])
            self.metadata.end_date = str(self.data.index[-1])
            self.metadata.variables = list(self.data.columns)

        except Exception as e:
            raise IOError(f"Failed to read CSV file {self.filepath}: {e}")

        return self.data

    def read_excel(self, sheet_name: Union[str, int] = 0,
                   date_column: str = "date") -> pd.DataFrame:
        """Read time series from Excel file.

        Args:
            sheet_name: Sheet name or index.
            date_column: Name of the date column.

        Returns:
            DataFrame with datetime index.
        """
        try:
            df = pd.read_excel(self.filepath, sheet_name=sheet_name)
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df = df.set_index(date_column)

            self.data = df
            self.metadata.start_date = str(df.index[0])
            self.metadata.end_date = str(df.index[-1])
            self.metadata.variables = list(df.columns)

        except Exception as e:
            raise IOError(f"Failed to read Excel file {self.filepath}: {e}")

        return self.data

    def read_netcdf(self, variable: str = "precipitation",
                    x_dim: str = "x", y_dim: str = "y",
                    time_dim: str = "time") -> Dict[str, np.ndarray]:
        """Read time series from NetCDF file.

        Args:
            variable: Variable name to read.
            x_dim: X dimension name.
            y_dim: Y dimension name.
            time_dim: Time dimension name.

        Returns:
            Dictionary with 'data', 'lons', 'lats', 'times'.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray is required for NetCDF reading. Install with: pip install xarray")

        try:
            ds = xr.open_dataset(self.filepath)

            result = {
                'data': ds[variable].values,
                'times': ds[time_dim].values,
            }

            if x_dim in ds.coords:
                result['lons'] = ds[x_dim].values
            if y_dim in ds.coords:
                result['lats'] = ds[y_dim].values

            self.data = ds[variable]
            self.metadata.variables = [variable]

            ds.close()

        except Exception as e:
            raise IOError(f"Failed to read NetCDF file {self.filepath}: {e}")

        return result

    def resample(self, freq: str = "D") -> pd.DataFrame:
        """Resample time series to new frequency.

        Args:
            freq: Target frequency ('D' for daily, 'H' for hourly, etc.).

        Returns:
            Resampled DataFrame.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read_csv or read_excel first.")

        return self.data.resample(freq).mean()

    def to_numpy(self, columns: Optional[List[str]] = None) -> np.ndarray:
        """Convert DataFrame to numpy array.

        Args:
            columns: Columns to extract. If None, uses all.

        Returns:
            Numpy array.
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        if columns:
            return self.data[columns].values
        return self.data.values


class BasinForcingReader:
    """Reader for basin-scale forcing data (P, PET, T, etc.).

    Designed for CAMELS-style datasets.
    """

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.basins: List[str] = []
        self._cache: Dict[str, pd.DataFrame] = {}

    def discover_basins(self) -> List[str]:
        """Discover available basins from subdirectories."""
        if not self.data_dir.exists():
            return []

        self.basins = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        return sorted(self.basins)

    def load_basin(self, basin_id: str,
                   forcing_file: str = "forcing.csv",
                   date_column: str = "date") -> pd.DataFrame:
        """Load forcing data for a specific basin.

        Args:
            basin_id: Basin identifier.
            forcing_file: Name of forcing file.
            date_column: Date column name.

        Returns:
            DataFrame with forcing data.
        """
        if basin_id in self._cache:
            return self._cache[basin_id]

        basin_dir = self.data_dir / basin_id
        forcing_path = basin_dir / forcing_file

        if not forcing_path.exists():
            raise FileNotFoundError(f"Forcing file not found: {forcing_path}")

        reader = TimeSeriesReader(forcing_path)
        df = reader.read_csv(date_column=date_column)

        self._cache[basin_id] = df
        return df

    def load_multiple_basins(self, basin_ids: List[str],
                            forcing_file: str = "forcing.csv") -> Dict[str, pd.DataFrame]:
        """Load forcing data for multiple basins.

        Args:
            basin_ids: List of basin identifiers.
            forcing_file: Name of forcing file.

        Returns:
            Dictionary mapping basin_id to DataFrame.
        """
        result = {}
        for basin_id in basin_ids:
            result[basin_id] = self.load_basin(basin_id, forcing_file)
        return result

    def get_variable(self, basin_id: str, variable: str) -> np.ndarray:
        """Get a specific variable for a basin.

        Args:
            basin_id: Basin identifier.
            variable: Variable name.

        Returns:
            Array of variable values.
        """
        df = self.load_basin(basin_id)
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found. Available: {df.columns.tolist()}")
        return df[variable].values


def read_forcing(filepath: Union[str, Path],
                format: str = "auto",
                **kwargs) -> pd.DataFrame:
    """Convenience function to read forcing data.

    Args:
        filepath: Path to forcing file.
        format: File format ('csv', 'excel', 'netcdf', or 'auto').
        **kwargs: Additional arguments for specific readers.

    Returns:
        DataFrame with time series data.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if format == "auto":
        if suffix in ['.csv']:
            format = "csv"
        elif suffix in ['.xlsx', '.xls']:
            format = "excel"
        elif suffix in ['.nc']:
            format = "netcdf"

    reader = TimeSeriesReader(path)

    if format == "csv" or suffix in ['.csv']:
        return reader.read_csv(**kwargs)
    elif format == "excel":
        return reader.read_excel(**kwargs)
    elif format == "netcdf":
        result = reader.read_netcdf(**kwargs)
        return result['data']
    else:
        raise ValueError(f"Unknown format: {format}")
