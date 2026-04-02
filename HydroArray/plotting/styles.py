"""
顶刊风格配色方案
Inspired by: Nature, Science, JGR, HESS, WRR, GRL, Earth-Science Reviews
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── 配色色板 ──────────────────────────────────────────────────────────────────

class C:
    # 蓝系（水文/大气/海洋）
    NATURE_BLUE   = ["#3B4992", "#4A90D9", "#7BAFD4", "#A8D8EA", "#1F77B4"]
    # 暖色系（气候变化/能源）
    NATURE_WARM   = ["#C05A2D", "#E07B39", "#F4A261", "#E9C46A", "#8B5E3C"]
    # 绿系（生态/陆地）
    NATURE_GREEN  = ["#2D6A4F", "#40916C", "#74C69D", "#95D5B2", "#1B4332"]
    # Nature主色盘
    NATURE_PUB    = ["#C13A3A", "#E07B39", "#3B7EA1", "#5A8F3E", "#8B5CF6", "#E89AC7", "#4A4A4A"]
    # 地球物理/水文学（蓝色渐变）
    JGR_BLUE      = ["#003F5C", "#0868AC", "#3690C0", "#74A9DE", "#A6BDDB", "#D0D1E6"]
    # HESS/WRR 水文风格（蓝绿+土色）
    HESS_BLUE     = ["#1A5276", "#2980B9", "#7FB3D3", "#1ABC9C", "#F39C12", "#E74C3C"]
    # 深色专业（海报/论文插图）
    DARK_PRO      = ["#00B4D8", "#48CAE4", "#90E0EF", "#CAF0F8", "#FF6B6B", "#C77DFF"]
    # 彩虹（水文极端值）
    RAINBOW       = ["#3B4992", "#00B4D8", "#76C893", "#F4D03F", "#E76F51", "#9B2226"]
    # 灰色+强调色
    GRAY_ACCENT   = ["#2C3E50", "#7F8C8D", "#95A5A6", "#BDC3C7", "#E74C3C", "#3498DB"]

    # ── 颜色映射 (Colormaps) ────────────────────────────────────────────────
    # 降水配色：白 → 蓝 → 绿 → 黄 → 红 → 紫
    PRECIP = [
        (1.0, 1.0, 1.0, 0.0),
        (0.7, 0.9, 1.0, 1.0),
        (0.4, 0.7, 1.0, 1.0),
        (0.2, 0.5, 0.9, 1.0),
        (0.0, 0.8, 0.4, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 0.6, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.8, 0.0, 0.4, 1.0),
        (0.6, 0.0, 0.6, 1.0),
    ]

    # 温度配色：蓝 → 白 → 红
    TEMPERATURE = [
        (0.0, 0.0, 0.8, 1.0),
        (0.3, 0.5, 1.0, 1.0),
        (0.7, 0.85, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 0.9, 0.7, 1.0),
        (1.0, 0.5, 0.3, 1.0),
        (0.8, 0.0, 0.0, 1.0),
    ]

    # 地形配色：深蓝 → 浅蓝 → 绿 → 黄 → 棕 → 白
    TERRAIN = [
        (0.0, 0.1, 0.3, 1.0),
        (0.2, 0.4, 0.7, 1.0),
        (0.4, 0.7, 0.9, 1.0),
        (0.3, 0.7, 0.3, 1.0),
        (0.6, 0.8, 0.3, 1.0),
        (0.8, 0.7, 0.4, 1.0),
        (0.6, 0.4, 0.2, 1.0),
        (0.9, 0.9, 0.85, 1.0),
        (1.0, 1.0, 1.0, 1.0),
    ]

    # 土壤湿度：棕 → 黄 → 绿 → 蓝
    SOIL_MOISTURE = [
        (0.4, 0.2, 0.1, 1.0),
        (0.6, 0.4, 0.2, 1.0),
        (0.8, 0.7, 0.3, 1.0),
        (0.7, 0.8, 0.4, 1.0),
        (0.4, 0.7, 0.5, 1.0),
        (0.3, 0.6, 0.7, 1.0),
        (0.2, 0.4, 0.8, 1.0),
    ]

    # 蒸散发：白 → 浅蓝 → 绿 → 深绿
    ET = [
        (1.0, 1.0, 1.0, 1.0),
        (0.9, 0.95, 1.0, 1.0),
        (0.7, 0.9, 0.95, 1.0),
        (0.5, 0.85, 0.8, 1.0),
        (0.3, 0.75, 0.6, 1.0),
        (0.15, 0.6, 0.45, 1.0),
        (0.05, 0.45, 0.3, 1.0),
    ]

    # 径流：白 → 浅蓝 → 蓝 → 深蓝
    RUNOFF = [
        (1.0, 1.0, 1.0, 1.0),
        (0.85, 0.92, 1.0, 1.0),
        (0.6, 0.8, 0.95, 1.0),
        (0.4, 0.65, 0.9, 1.0),
        (0.25, 0.5, 0.85, 1.0),
        (0.15, 0.35, 0.75, 1.0),
        (0.08, 0.2, 0.55, 1.0),
    ]


# ── 风格预设 ─────────────────────────────────────────────────────────────────

STYLES = {

    "nature": {
        "name": "Nature / Science",
        "colors": C.NATURE_BLUE,
        "rc": {
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.prop_cycle": plt.cycler(color=C.NATURE_BLUE),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    },

    "hess": {
        "name": "Hydrology & Earth System (HESS/WRR)",
        "colors": C.HESS_BLUE,
        "rc": {
            'font.family': ['Times New Roman', 'SimSun'],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "axes.prop_cycle": plt.cycler(color=C.HESS_BLUE),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    },

    "jgr": {
        "name": "JGR Atmospheres / 地球物理蓝",
        "colors": C.JGR_BLUE,
        "rc": {
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": "-",
            "axes.prop_cycle": plt.cycler(color=C.JGR_BLUE),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    },

    "dark_pro": {
        "name": "深色专业（海报/演示）",
        "colors": C.DARK_PRO,
        "rc": {
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "--",
            "axes.facecolor": "#0D1117",
            "figure.facecolor": "#0D1117",
            "axes.edgecolor": "#30363D",
            "axes.labelcolor": "#E6EDF3",
            "xtick.color": "#8B949E",
            "ytick.color": "#8B949E",
            "text.color": "#E6EDF3",
            "axes.prop_cycle": plt.cycler(color=C.DARK_PRO),
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    },

    "warm_climate": {
        "name": "气候变化暖色调",
        "colors": C.NATURE_WARM,
        "rc": {
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.prop_cycle": plt.cycler(color=C.NATURE_WARM),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    },

    "eco_green": {
        "name": "生态绿（陆地/植被水文）",
        "colors": C.NATURE_GREEN,
        "rc": {
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.prop_cycle": plt.cycler(color=C.NATURE_GREEN),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    },

    "nature_pub": {
        "name": "Nature 官方配色（最多彩）",
        "colors": C.NATURE_PUB,
        "rc": {
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.prop_cycle": plt.cycler(color=C.NATURE_PUB),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    },
}


def use_hydro_style(style: str = "hess", chinese: bool = True, dpi: int = 300) -> None:
    """
    应用水文风格

    参数:
        style: 风格名，可选: nature / hess / jgr / dark_pro / warm_climate / eco_green / nature_pub
        chinese: 是否启用中文支持（会用 DejaVu Sans，fallback 不保证中文渲染）
        dpi: 导出分辨率
    """
    if style not in STYLES:
        raise ValueError(f"未知的风格 '{style}'，可选: {list(STYLES.keys())}")

    cfg = STYLES[style]
    rc = cfg["rc"].copy()
    rc["figure.dpi"] = dpi
    rc["savefig.dpi"] = dpi

    plt.rcParams.update(rc)


def get_colors(style: str) -> list:
    """获取风格对应的颜色列表"""
    if style not in STYLES:
        raise ValueError(f"未知的风格 '{style}'，可选: {list(STYLES.keys())}")
    return STYLES[style]["colors"]


def list_styles() -> dict:
    """返回所有风格的 {key: description} 字典"""
    return {k: v["name"] for k, v in STYLES.items()}


def apply_theme(style: str = "hess") -> None:
    """
    应用主题（apply_theme 是 use_hydro_style 的别名）

    参数:
        style: 风格名，可选: nature / hess / jgr / dark_pro / warm_climate / eco_green / nature_pub
    """
    use_hydro_style(style=style)


# ── 颜色映射名称映射 ───────────────────────────────────────────────────────────

COLORMAP_ALIASES = {
    "precip": "PRECIP",
    "precipitation": "PRECIP",
    "rain": "PRECIP",
    "temp": "TEMPERATURE",
    "temperature": "TEMPERATURE",
    "terrain": "TERRAIN",
    "elevation": "TERRAIN",
    "soil": "SOIL_MOISTURE",
    "soil_moisture": "SOIL_MOISTURE",
    "moisture": "SOIL_MOISTURE",
    "et": "ET",
    "evapotranspiration": "ET",
    "runoff": "RUNOFF",
    "discharge": "RUNOFF",
    "flow": "RUNOFF",
    "hydro": "JGR_BLUE",
}


def get_colormap(name: str) -> "mcolors.Colormap":
    """
    获取颜色映射

    Args:
        name: 颜色映射名称，支持：
            - 内置水文配色: "precip", "temperature", "terrain", "soil_moisture", "et", "runoff", "hydro"
            - matplotlib 内置: "Blues", "Reds", "viridis", "RdYlBu" 等

    Returns:
        matplotlib Colormap 对象

    Examples:
        >>> import HydroArray as ha
        >>> cmap = ha.get_colormap("precip")
        >>> cmap = ha.get_colormap("Blues")
    """
    name_lower = name.lower()
    
    if name_lower in COLORMAP_ALIASES:
        attr_name = COLORMAP_ALIASES[name_lower]
        colors = getattr(C, attr_name)
        return mcolors.LinearSegmentedColormap.from_list(name_lower, colors, N=256)
    
    return plt.get_cmap(name)


def list_colormaps() -> dict:
    """
    列出所有可用的颜色映射

    Returns:
        {名称: 描述} 字典
    """
    built_in = {
        "precip": "降水（白→蓝→绿→黄→红→紫）",
        "temperature": "温度（蓝→白→红）",
        "terrain": "地形（深蓝→浅蓝→绿→黄→棕→白）",
        "soil_moisture": "土壤湿度（棕→黄→绿→蓝）",
        "et": "蒸散发（白→浅蓝→绿→深绿）",
        "runoff": "径流（白→浅蓝→蓝→深蓝）",
        "hydro": "水文蓝（JGR风格蓝色渐变）",
    }
    return built_in
