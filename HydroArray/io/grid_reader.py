"""
Grid Data Reader for Traditional Hydrological Models

Supports reading raster data in ASCII Grid and GeoTIFF formats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class GridMetadata:
    """Metadata for grid data."""
    nrows: int = 0
    ncols: int = 0
    xllcorner: float = 0.0
    yllcorner: float = 0.0
    cellsize: float = 1.0
    nodata: float = -9999.0
    projection: Optional[str] = None
    crs: str = "EPSG:4326"


class ASCGridReader:
    """Reader for ESRI ASCII Grid format (.asc).

    Format:
        ncols         360
        nrows         180
        xllcorner     -180.0
        yllcorner     -90.0
        cellsize      1.0
        NODATA_value  -9999
        data values...
    """

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.metadata = GridMetadata()
        self.data: Optional[np.ndarray] = None

    def read(self) -> Tuple[np.ndarray, GridMetadata]:
        """Read ASCII grid file.

        Returns:
            Tuple of (data array, metadata).
            Data array shape is (nrows, ncols).
        """
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # Parse header
        header = {}
        data_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) == 2:
                try:
                    header[parts[0].lower()] = float(parts[1])
                except ValueError:
                    header[parts[0].lower()] = parts[1]
            else:
                # Data line
                data_lines.append([float(v) for v in parts])

        # Extract metadata
        self.metadata.ncols = int(header.get('ncols', 0))
        self.metadata.nrows = int(header.get('nrows', 0))
        self.metadata.xllcorner = header.get('xllcorner', header.get('xllcenter', 0))
        self.metadata.yllcorner = header.get('yllcorner', header.get('yllcenter', 0))
        self.metadata.cellsize = header.get('cellsize', 1.0)
        self.metadata.nodata = header.get('nodata_value', -9999)

        # Parse data
        if data_lines:
            self.data = np.array(data_lines)
        else:
            # Data might start after header without blank line
            data_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= self.metadata.ncols:
                    try:
                        data_lines.append([float(v) for v in parts[:self.metadata.ncols]])
                    except ValueError:
                        continue

            if data_lines:
                self.data = np.array(data_lines)

        # Handle row order (ASC files start from top row)
        if self.data is not None and self.data.shape[0] == self.metadata.nrows:
            self.data = np.flipud(self.data)

        return self.data, self.metadata

    def get_lats_lons(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get latitude and longitude arrays.

        Returns:
            Tuple of (lats, lons) arrays.
        """
        if self.data is None:
            self.read()

        lats = np.linspace(
            self.metadata.yllcorner + self.metadata.cellsize * self.metadata.nrows,
            self.metadata.yllcorner,
            self.metadata.nrows
        )
        lons = np.linspace(
            self.metadata.xllcorner,
            self.metadata.xllcorner + self.metadata.cellsize * self.metadata.ncols,
            self.metadata.ncols
        )

        return lats, lons


class GeoTIFFReader:
    """Reader for GeoTIFF format (.tif, .tiff).

    Requires rasterio package.
    """

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.metadata = GridMetadata()
        self.data: Optional[np.ndarray] = None
        self._rasterio = None

    def _import_rasterio(self):
        """Lazy import of rasterio."""
        if self._rasterio is None:
            try:
                import rasterio
                self._rasterio = rasterio
            except ImportError:
                raise ImportError(
                    "rasterio is required for GeoTIFF reading. "
                    "Install with: pip install rasterio"
                )
        return self._rasterio

    def read(self) -> Tuple[np.ndarray, GridMetadata]:
        """Read GeoTIFF file.

        Returns:
            Tuple of (data array, metadata).
        """
        rio = self._import_rasterio()

        with rio.open(self.filepath) as src:
            self.data = src.read(1)

            self.metadata.nrows, self.metadata.ncols = self.data.shape
            self.metadata.cellsize = src.transform[0]

            # Get bounds
            bounds = src.bounds
            self.metadata.xllcorner = bounds.left
            self.metadata.yllcorner = bounds.bottom

            # Get nodata value
            self.metadata.nodata = src.nodata if src.nodata is not None else -9999

            # Get CRS
            if src.crs is not None:
                self.metadata.crs = str(src.crs)
                self.metadata.projection = src.crs.to_wkt()

        # Handle nodata
        if self.data is not None:
            self.data = np.where(self.data == self.metadata.nodata, np.nan, self.data)

        return self.data, self.metadata

    def get_lats_lons(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get latitude and longitude arrays."""
        if self.data is None:
            self.read()

        rio = self._import_rasterio()

        with rio.open(self.filepath) as src:
            transform = src.transform

            rows, cols = self.data.shape
            lons = np.zeros(cols)
            lats = np.zeros(rows)

            for c in range(cols):
                x = transform[2] + c * transform[0] + transform[0] / 2
                lons[c] = x

            for r in range(rows):
                y = transform[5] + r * transform[4] + transform[4] / 2
                lats[r] = y

        return lats, lons


class GridStackReader:
    """Reader for multiple grids forming a time series.

    Reads grids from a directory, assuming files are named with timestamps.
    """

    def __init__(self, directory: Union[str, Path],
                 pattern: str = "*.asc",
                 reader_class=ASCGridReader):
        self.directory = Path(directory)
        self.pattern = pattern
        self.reader_class = reader_class
        self.files: List[Path] = []
        self.timestamps: List[str] = []
        self._data_cache: Dict[str, np.ndarray] = {}

    def discover_files(self) -> List[Path]:
        """Discover grid files in directory.

        Returns:
            List of file paths sorted by name.
        """
        self.files = sorted(self.directory.glob(self.pattern))

        if not self.files:
            # Try other common patterns
            alt_patterns = ["*.tif", "*.tiff", "*.asc", "*.ascii"]
            for p in alt_patterns:
                self.files = sorted(self.directory.glob(p))
                if self.files:
                    break

        return self.files

    def read_at_index(self, index: int) -> Tuple[np.ndarray, GridMetadata]:
        """Read grid at specific index.

        Args:
            index: File index.

        Returns:
            Tuple of (data, metadata).
        """
        if not self.files:
            self.discover_files()

        if index < 0 or index >= len(self.files):
            raise IndexError(f"Index {index} out of range for {len(self.files)} files")

        reader = self.reader_class(self.files[index])
        return reader.read()

    def read_time_range(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Read multiple grids as a time series.

        Args:
            start_idx: Starting index.
            end_idx: Ending index (inclusive).

        Returns:
            3D array with shape (n_times, nrows, ncols).
        """
        if not self.files:
            self.discover_files()

        n_times = end_idx - start_idx + 1
        first_data, metadata = self.read_at_index(start_idx)
        nrows, ncols = first_data.shape

        stack = np.zeros((n_times, nrows, ncols))
        stack[0] = first_data

        for i in range(start_idx + 1, end_idx + 1):
            data, _ = self.read_at_index(i)
            stack[i - start_idx] = data

        return stack

    def read_all(self) -> Tuple[np.ndarray, List[GridMetadata]]:
        """Read all grids in directory.

        Returns:
            Tuple of (3D data array, list of metadata).
        """
        if not self.files:
            self.discover_files()

        metadata_list = []
        data_list = []

        for f in self.files:
            reader = self.reader_class(f)
            data, meta = reader.read()
            data_list.append(data)
            metadata_list.append(meta)

        return np.array(data_list), metadata_list


def read_grid(filepath: Union[str, Path]) -> Tuple[np.ndarray, GridMetadata]:
    """Convenience function to read a grid file.

    Automatically detects format (ASC or GeoTIFF).

    Args:
        filepath: Path to grid file.

    Returns:
        Tuple of (data array, metadata).
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix in ['.asc', '.ascii']:
        reader = ASCGridReader(path)
    elif suffix in ['.tif', '.tiff']:
        reader = GeoTIFFReader(path)
    else:
        # Try ASC first
        try:
            reader = ASCGridReader(path)
            reader.read()
        except Exception:
            reader = GeoTIFFReader(path)

    return reader.read()
