"""
数据容器模块

提供统一的水文数据存储格式，支持多种数据类型转换。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False


@dataclass
class HydroData:
    """
    水文数据基类容器

    统一存储水文数据，支持多种输出格式转换。

    Attributes:
        data: 核心数据（numpy数组或字典）
        coords: 坐标信息 {'lat': ..., 'lon': ..., 'time': ...}
        dims: 维度名称列表
        attrs: 全局属性/元数据
        name: 数据名称
    """
    data: Union[np.ndarray, Dict[str, np.ndarray]]
    coords: Dict[str, np.ndarray] = field(default_factory=dict)
    dims: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        if isinstance(self.data, np.ndarray) and not self.dims:
            self.dims = [f"dim_{i}" for i in range(self.data.ndim)]

    def __len__(self) -> int:
        if isinstance(self.data, np.ndarray):
            return len(self.data)
        return len(self.data)

    def __repr__(self) -> str:
        data_type = "array" if isinstance(self.data, np.ndarray) else "dict"
        shape = self.data.shape if isinstance(self.data, np.ndarray) else f"{len(self.data)} vars"
        return f"HydroData(data={data_type}, shape={shape}, dims={self.dims})"

    def to_dataframe(self) -> "pd.DataFrame":
        if not HAS_PANDAS:
            raise ImportError("pandas 未安装，请运行: pip install pandas")

        if isinstance(self.data, dict):
            df = pd.DataFrame(self.data)
        else:
            if self.data.ndim == 1:
                df = pd.DataFrame({self.name or "value": self.data})
            elif self.data.ndim == 2:
                df = pd.DataFrame(self.data)
            else:
                flat_data = self.data.reshape(-1)
                df = pd.DataFrame({self.name or "value": flat_data})

        for coord_name, coord_vals in self.coords.items():
            if len(coord_vals) == len(df):
                df[coord_name] = coord_vals

        for key, val in self.attrs.items():
            df.attrs[key] = val

        return df

    def to_xarray(self) -> "xr.DataArray":
        if not HAS_XARRAY:
            raise ImportError("xarray 未安装，请运行: pip install xarray")

        if isinstance(self.data, dict):
            raise ValueError("字典数据无法直接转换为 DataArray，请使用 to_dataset()")

        da = xr.DataArray(
            data=self.data,
            dims=self.dims,
            coords=self.coords,
            name=self.name,
            attrs=self.attrs
        )
        return da

    def to_dataset(self) -> "xr.Dataset":
        if not HAS_XARRAY:
            raise ImportError("xarray 未安装，请运行: pip install xarray")

        if isinstance(self.data, dict):
            data_vars = {}
            for key, val in self.data.items():
                if isinstance(val, np.ndarray):
                    data_vars[key] = (self.dims, val) if self.dims else val
            ds = xr.Dataset(data_vars, coords=self.coords, attrs=self.attrs)
        else:
            ds = self.to_xarray().to_dataset()

        return ds

    def to_numpy(self) -> np.ndarray:
        if isinstance(self.data, dict):
            raise ValueError("字典数据无法直接转换为单一数组")
        return self.data

    def to_list(self) -> Union[List, Dict[str, List]]:
        if isinstance(self.data, dict):
            return {k: v.tolist() for k, v in self.data.items()}
        return self.data.tolist()

    def to_zarr(self, path: Union[str, Path], **kwargs) -> None:
        if not HAS_ZARR:
            raise ImportError("zarr 未安装，请运行: pip install zarr")

        ds = self.to_dataset()
        ds.to_zarr(path, **kwargs)

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame",
                       data_cols: Optional[List[str]] = None,
                       coord_cols: Optional[List[str]] = None,
                       name: str = "") -> "HydroData":
        coord_cols = coord_cols or []
        data_cols = data_cols or [c for c in df.columns if c not in coord_cols]

        coords = {c: df[c].values for c in coord_cols if c in df.columns}

        if len(data_cols) == 1:
            data = df[data_cols[0]].values
        else:
            data = {c: df[c].values for c in data_cols}

        dims = list(coords.keys()) if coords else ['index']
        attrs = dict(df.attrs) if hasattr(df, 'attrs') else {}

        return cls(data=data, coords=coords, dims=dims, attrs=attrs, name=name)

    @classmethod
    def from_xarray(cls, da_or_ds: Union["xr.DataArray", "xr.Dataset"],
                    name: str = "") -> "HydroData":
        if isinstance(da_or_ds, xr.DataArray):
            return cls(
                data=da_or_ds.values,
                coords={k: v.values for k, v in da_or_ds.coords.items()},
                dims=list(da_or_ds.dims),
                attrs=dict(da_or_ds.attrs),
                name=da_or_ds.name or name
            )
        else:
            data = {k: v.values for k, v in da_or_ds.data_vars.items()}
            return cls(
                data=data,
                coords={k: v.values for k, v in da_or_ds.coords.items()},
                dims=list(da_or_ds.dims),
                attrs=dict(da_or_ds.attrs),
                name=name
            )

    @classmethod
    def from_zarr(cls, path: Union[str, Path],
                  group: Optional[str] = None,
                  name: str = "") -> "HydroData":
        if not HAS_XARRAY:
            raise ImportError("xarray 未安装")

        ds = xr.open_zarr(path, group=group)
        return cls.from_xarray(ds, name=name)

    @classmethod
    def from_numpy(cls, arr: np.ndarray,
                   dims: Optional[List[str]] = None,
                   coords: Optional[Dict[str, np.ndarray]] = None,
                   name: str = "") -> "HydroData":
        return cls(
            data=arr,
            coords=coords or {},
            dims=dims or [],
            name=name
        )

    @classmethod
    def from_list(cls, data: List,
                  dims: Optional[List[str]] = None,
                  coords: Optional[Dict[str, np.ndarray]] = None,
                  name: str = "") -> "HydroData":
        return cls(
            data=np.array(data),
            coords=coords or {},
            dims=dims or [],
            name=name
        )


@dataclass
class GriddedData(HydroData):
    """
    网格数据容器

    专门用于卫星数据、再分析数据等网格化数据。

    Attributes:
        crs: 坐标参考系统
    """
    crs: str = "EPSG:4326"

    def __post_init__(self):
        super().__post_init__()
        if 'lat' in self.coords:
            self.lat = self.coords['lat']
        if 'lon' in self.coords:
            self.lon = self.coords['lon']
        if 'time' in self.coords:
            self.time = self.coords['time']

    def sel(self, lat: Optional[slice] = None,
            lon: Optional[slice] = None,
            time: Optional[slice] = None) -> "GriddedData":
        da = self.to_xarray()

        sel_dict = {}
        if lat is not None:
            sel_dict['lat'] = lat
        if lon is not None:
            sel_dict['lon'] = lon
        if time is not None:
            sel_dict['time'] = time

        if sel_dict:
            da = da.sel(sel_dict)

        return GriddedData.from_xarray(da)

    def to_geotiff(self, path: Union[str, Path],
                   time_index: Optional[int] = None) -> None:
        try:
            import rioxarray
        except ImportError:
            raise ImportError("rioxarray 未安装，请运行: pip install rioxarray")

        da = self.to_xarray()
        if time_index is not None and 'time' in da.dims:
            da = da.isel(time=time_index)

        da.rio.write_crs(self.crs, inplace=True)
        da.rio.to_raster(path)


@dataclass
class StationData(HydroData):
    """
    站点数据容器

    专门用于水文站点观测数据。

    Attributes:
        station_id: 站点ID列表
        station_name: 站点名称列表
        variables: 变量名称列表
    """
    station_id: Optional[List[str]] = None
    station_name: Optional[List[str]] = None
    variables: List[str] = field(default_factory=list)

    def get_station(self, station_id: str) -> "pd.DataFrame":
        df = self.to_dataframe()
        if 'station_id' in df.columns:
            return df[df['station_id'] == station_id]
        return df

    def to_camels_format(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()

        if self.station_id and 'station_id' in df.columns:
            for sid in self.station_id:
                station_df = df[df['station_id'] == sid]
                station_df.to_csv(path / f"{sid}.csv", index=False)
        else:
            df.to_csv(path / "data.csv", index=False)


def as_hydrodata(data: Any, **kwargs) -> HydroData:
    """
    将任意数据转换为 HydroData

    Args:
        data: 输入数据（DataFrame, DataArray, Dataset, numpy, list）
        **kwargs: 传递给相应构造函数的参数

    Returns:
        HydroData: 数据容器
    """
    if isinstance(data, HydroData):
        return data

    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        return HydroData.from_dataframe(data, **kwargs)

    if HAS_XARRAY and isinstance(data, (xr.DataArray, xr.Dataset)):
        return HydroData.from_xarray(data, **kwargs)

    if isinstance(data, np.ndarray):
        return HydroData.from_numpy(data, **kwargs)

    if isinstance(data, (list, tuple)):
        return HydroData.from_list(data, **kwargs)

    if isinstance(data, dict):
        return HydroData(data=data, **kwargs)

    raise TypeError(f"不支持的数据类型: {type(data)}")
