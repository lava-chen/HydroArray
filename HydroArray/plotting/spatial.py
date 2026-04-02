"""
空间图

提供卫星数据、网格数据的可视化功能。
"""

from pathlib import Path
from typing import Literal, Optional, Union, Tuple, List
from datetime import datetime

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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import ColorbarBase
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from HydroArray.plotting.styles import use_hydro_style, C, get_colormap
from HydroArray.core.containers import HydroData, GriddedData


def satellite_plot(
    data: Union["pd.DataFrame", "xr.DataArray", "xr.Dataset", np.ndarray],
    lat_col: str = "lat",
    lon_col: str = "lon",
    value_col: str = "value",
    time_col: str = "datetime",
    time: Optional[Union[int, str, datetime]] = None,
    show_basemap: bool = False,
    cmap: str = "Blues",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 150,
    style: str = "hess",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional["plt.Axes"] = None,
    show: bool = True,
    **kwargs
) -> "plt.Figure":
    """
    绘制卫星数据栅格图

    Args:
        data: 输入数据，支持 DataFrame、DataArray、Dataset 或 numpy 数组
        lat_col: DataFrame 中纬度列名
        lon_col: DataFrame 中经度列名
        value_col: DataFrame 中数值列名
        time_col: DataFrame 中时间列名，默认 "datetime"
        time: 时间筛选，支持：
            - int: 时间索引（第几个时间点）
            - str/datetime: 具体时间值
            - None: 自动选择第一个时间点
        show_basemap: 是否显示底层地图（需要 cartopy）
        cmap: 颜色映射，默认 "Blues"，可选 "precip"、"hydro"
        vmin: 颜色范围最小值
        vmax: 颜色范围最大值
        title: 图表标题
        figsize: 图表大小
        dpi: 分辨率
        style: HydroArray 风格名称
        save_path: 保存路径
        ax: 已有的 axes 对象
        show: 是否立即显示图表，默认 True
        **kwargs: 传递给 pcolormesh 或 scatter 的参数

    Returns:
        matplotlib Figure 对象

    Examples:
        >>> import HydroArray as ha
        >>> df = ha.read_satellite_folder("data/fy3g")
        >>> 
        >>> # 绘制第一个时间点
        >>> ha.satellite_plot(df, time=0)
        >>> 
        >>> # 绘制指定时间
        >>> ha.satellite_plot(df, time="2024-07-01 03:00:00")
        >>> 
        >>> # 带底图
        >>> ha.satellite_plot(df, time=0, show_basemap=True, cmap="precip")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib 未安装，请运行: pip install matplotlib")

    use_hydro_style(style=style, dpi=dpi)

    lon, lat, values, selected_time = _extract_coords_values(
        data, lat_col, lon_col, value_col, time_col, time
    )

    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    if show_basemap and not HAS_CARTOPY:
        print("警告: cartopy 未安装，将不显示底层地图。请运行: pip install cartopy")
        show_basemap = False

    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = _create_figure_with_projection(show_basemap, figsize)

    if show_basemap:
        _add_basemap(ax, **kwargs)

    _is_xarray = HAS_XARRAY and isinstance(data, (xr.DataArray, xr.Dataset))
    if _is_xarray or _is_gridded(lon, lat):
        mesh = _plot_gridded(ax, lon, lat, values, cmap, vmin, vmax, show_basemap, **kwargs)
    else:
        mesh = _plot_scatter(ax, lon, lat, values, cmap, vmin, vmax, show_basemap, **kwargs)

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
    cbar.set_label(value_col)

    if show_basemap:
        _add_gridlines(ax)
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    if title is None and selected_time is not None:
        title = f"Time: {selected_time}"
    if title:
        ax.set_title(title)

    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _extract_coords_values(
    data: Union["pd.DataFrame", "xr.DataArray", "xr.Dataset", np.ndarray],
    lat_col: str,
    lon_col: str,
    value_col: str,
    time_col: str,
    time: Optional[Union[int, str, datetime]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    """从各种数据格式中提取坐标和数值"""
    selected_time = None

    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        if time_col in data.columns:
            unique_times = sorted(data[time_col].unique())
            
            if time is None:
                time = 0
            
            if isinstance(time, int):
                if time < 0:
                    time = len(unique_times) + time
                if time >= len(unique_times):
                    raise IndexError(f"时间索引 {time} 超出范围，共 {len(unique_times)} 个时间点")
                selected_time = unique_times[time]
            else:
                selected_time = time
            
            subset = data[data[time_col] == selected_time]
            lat = subset[lat_col].values
            lon = subset[lon_col].values
            values = subset[value_col].values
        else:
            lat = data[lat_col].values
            lon = data[lon_col].values
            values = data[value_col].values
        
        return lon, lat, values, str(selected_time) if selected_time else None

    if HAS_XARRAY and isinstance(data, xr.DataArray):
        if time is not None and "time" in data.dims:
            if isinstance(time, int):
                data = data.isel(time=time)
                selected_time = str(data.coords["time"].values)
            else:
                data = data.sel(time=time)
                selected_time = str(time)
        
        lat = data.coords["lat"].values if "lat" in data.coords else data.coords["latitude"].values
        lon = data.coords["lon"].values if "lon" in data.coords else data.coords["longitude"].values
        values = data.values
        
        if values.ndim == 2:
            lon, lat = np.meshgrid(lon, lat)
        
        return lon, lat, values, selected_time

    if HAS_XARRAY and isinstance(data, xr.Dataset):
        var_name = list(data.data_vars)[0]
        return _extract_coords_values(data[var_name], lat_col, lon_col, value_col, time_col, time)

    if isinstance(data, np.ndarray):
        raise ValueError("numpy 数组需要提供坐标信息，请使用 DataFrame 或 DataArray")

    raise TypeError(f"不支持的数据类型: {type(data)}")


def _is_gridded(lon: np.ndarray, lat: np.ndarray) -> bool:
    """判断数据是否为网格格式"""
    return lon.ndim == 2 and lat.ndim == 2


def _create_figure_with_projection(show_basemap: bool, figsize: Tuple[float, float]):
    """创建带有投影的图表"""
    if show_basemap:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _add_basemap(ax: "plt.Axes", **kwargs):
    """添加底层地图要素"""
    resolution = kwargs.pop("resolution", "110m")
    land_color = kwargs.pop("land_color", "#f5f5f5")
    ocean_color = kwargs.pop("ocean_color", "#e6f3ff")
    coastline_color = kwargs.pop("coastline_color", "#333333")
    borders_color = kwargs.pop("borders_color", "#666666")

    ax.add_feature(cfeature.LAND.with_scale(resolution), facecolor=land_color)
    ax.add_feature(cfeature.OCEAN.with_scale(resolution), facecolor=ocean_color)
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), linewidth=0.5, color=coastline_color)
    ax.add_feature(cfeature.BORDERS.with_scale(resolution), linewidth=0.3, linestyle="--", color=borders_color)

    rivers = kwargs.pop("rivers", True)
    if rivers:
        ax.add_feature(cfeature.RIVERS.with_scale(resolution), linewidth=0.3, color="#4a90d9")


def _add_gridlines(ax: "plt.Axes"):
    """添加经纬网格线"""
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def _plot_gridded(
    ax: "plt.Axes",
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    show_basemap: bool,
    **kwargs
) -> "plt.cm.ScalarMappable":
    """绘制网格数据"""
    cmap_obj = _get_colormap(cmap)

    transform = ccrs.PlateCarree() if show_basemap else None

    mesh = ax.pcolormesh(
        lon, lat, values,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        transform=transform,
        shading="auto",
        **kwargs
    )

    if show_basemap:
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=transform)

    return mesh


def _plot_scatter(
    ax: "plt.Axes",
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    show_basemap: bool,
    **kwargs
) -> "plt.cm.ScalarMappable":
    """绘制散点数据"""
    cmap_obj = _get_colormap(cmap)
    
    s = kwargs.pop("s", 1)
    alpha = kwargs.pop("alpha", 0.8)
    marker = kwargs.pop("marker", "s")

    transform = ccrs.PlateCarree() if show_basemap else None

    scatter_kwargs = {"c": values, "cmap": cmap_obj, "vmin": vmin, "vmax": vmax, "s": s, "alpha": alpha, "marker": marker}
    if transform:
        scatter_kwargs["transform"] = transform
    scatter_kwargs.update(kwargs)

    mesh = ax.scatter(lon, lat, **scatter_kwargs)

    if show_basemap:
        pad = 0.5
        ax.set_extent([lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad], crs=transform)

    return mesh


def _get_colormap(cmap: str) -> "mcolors.Colormap":
    """获取颜色映射（委托给 styles.get_colormap）"""
    return get_colormap(cmap)


def get_available_times(
    data: Union["pd.DataFrame", "xr.DataArray", "xr.Dataset", np.ndarray, HydroData, GriddedData],
    time_col: str = "datetime"
) -> List:
    """
    获取数据中所有可用的时间点

    Args:
        data: 输入数据（DataFrame, DataArray, Dataset, HydroData, GriddedData）
        time_col: 时间列名

    Returns:
        时间点列表

    Examples:
        >>> import HydroArray as ha
        >>> df = ha.read_satellite_folder("data/fy3g")
        >>> times = ha.get_available_times(df)
        >>> print(f"共 {len(times)} 个时间点")
        >>> print(times[:5])  # 显示前5个
    """
    if isinstance(data, (HydroData, GriddedData)):
        if "time" in data.coords:
            return list(data.coords["time"])
        return []
    
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        if time_col in data.columns:
            return sorted(data[time_col].unique().tolist())
        return []
    
    if HAS_XARRAY and isinstance(data, xr.DataArray):
        if "time" in data.dims:
            return data.coords["time"].values.tolist()
        return []
    
    if HAS_XARRAY and isinstance(data, xr.Dataset):
        if "time" in data.dims:
            return data.coords["time"].values.tolist()
        return []
    
    raise TypeError(f"不支持的数据类型: {type(data)}")


def satellite_animation(
    data: Union["pd.DataFrame", "xr.DataArray"],
    output_folder: Union[str, Path],
    lat_col: str = "lat",
    lon_col: str = "lon",
    value_col: str = "value",
    time_col: str = "datetime",
    fps: int = 5,
    **kwargs
) -> str:
    """
    生成卫星数据动画

    Args:
        data: 输入数据
        output_folder: 输出文件夹
        lat_col: 纬度列名
        lon_col: 经度列名
        value_col: 数值列名
        time_col: 时间列名
        fps: 帧率
        **kwargs: 传递给 satellite_plot 的参数

    Returns:
        生成的动画文件路径
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio 未安装，请运行: pip install imageio")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    times = get_available_times(data, time_col)
    
    if not times:
        raise ValueError("数据中没有时间信息")

    images = []
    for i, t in enumerate(times):
        fig_path = output_folder / f"frame_{i:04d}.png"
        satellite_plot(
            data,
            lat_col=lat_col,
            lon_col=lon_col,
            value_col=value_col,
            time_col=time_col,
            time=t,
            title=f"Time: {t}",
            save_path=fig_path,
            show=False,
            **kwargs
        )
        plt.close()
        images.append(imageio.imread(fig_path))

    gif_path = output_folder / "animation.gif"
    imageio.mimsave(gif_path, images, fps=fps)

    print(f"动画已保存到: {gif_path}")
    return str(gif_path)
