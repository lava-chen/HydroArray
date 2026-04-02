"""
文件名解析工具

从文件名和文件夹名中提取时间和数据源信息。
支持多种常见的卫星数据和水文数据命名格式。
"""

import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ParsedFileInfo:
    """
    解析后的文件信息

    Attributes:
        datetime: 时间戳
        source: 数据源名称（如 fy3g, gpm, era5）
        date: 日期字符串（YYYYMMDD格式）
        time: 时间字符串（HH格式）
        original_name: 原始文件名
    """
    datetime: Optional[datetime] = None
    source: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    original_name: str = ""


SOURCE_ALIAS_MAP = {
    'FY3G': 'fy3g',
    'FY3': 'fy3',
    'FY4': 'fy4',
    'GPM': 'gpm',
    'IMERG': 'gpm',
    'IMERG_E': 'gpm',
    'IMERG_L': 'gpm',
    'TRMM': 'trmm',
    'ERA5': 'era5',
    'ERA': 'era5',
    'CMORPH': 'cmorph',
    'PERSIANN': 'persiann',
    'CHIRPS': 'chirps',
    'MSWEP': 'mswep',
    'CPC': 'cpc',
    'GPCC': 'gpcc',
    'CRU': 'cru',
    'CLDAS': 'cldas',
    'GLDAS': 'gldas',
    'NLDAS': 'nldas',
    'MERRA': 'merra',
    'MERRA2': 'merra2',
    'JRA55': 'jra55',
    'CFSR': 'cfsr',
    'CFSV2': 'cfsv2',
    'HYCOM': 'hycom',
    'ORAS5': 'oras5',
    'SODA': 'soda',
    'WOA': 'woa',
    'CAMELS': 'camels',
    'GSIM': 'gsim',
    'GRDC': 'grdc',
}

FILENAME_PATTERNS = [
    (re.compile(r'^([A-Za-z0-9]+)[_-](\d{8})[_-](\d{2})(?:[_-].*)?\.csv$', re.IGNORECASE), 'source_date_time'),
    (re.compile(r'^([A-Za-z0-9]+)[_-](\d{8})[_-](\d{4})(?:[_-].*)?\.csv$', re.IGNORECASE), 'source_date_hhmm'),
    (re.compile(r'^([A-Za-z0-9]+)[_-](\d{8})(?:[_-].*)?\.csv$', re.IGNORECASE), 'source_date'),
    (re.compile(r'^(\d{8})[_-](\d{2})[_-]([A-Za-z0-9]+)(?:[_-].*)?\.csv$', re.IGNORECASE), 'date_time_source'),
    (re.compile(r'^(\d{8})[_-]([A-Za-z0-9]+)(?:[_-].*)?\.csv$', re.IGNORECASE), 'date_source'),
    (re.compile(r'^(\d{4})[/-](\d{2})[/-](\d{2})[_-](\d{2})[_-]([A-Za-z0-9]+)(?:[_-].*)?\.csv$', re.IGNORECASE), 'ymd_time_source'),
    (re.compile(r'^([A-Za-z0-9]+)[_-](\d{4})[/-](\d{2})[/-](\d{2})[_-](\d{2})(?:[_-].*)?\.csv$', re.IGNORECASE), 'source_ymd_time'),
    (re.compile(r'^(\d{4})(\d{2})(\d{2})[_-](\d{2})[_-]([A-Za-z0-9]+)(?:[_-].*)?\.csv$', re.IGNORECASE), 'yyyymmdd_time_source'),
    (re.compile(r'^([A-Za-z0-9]+)[_-](\d{4})(\d{2})(\d{2})[_-](\d{2})(?:[_-].*)?\.csv$', re.IGNORECASE), 'source_yyyymmdd_time'),
]

FOLDER_PATTERNS = [
    re.compile(r'(fy3g|fy3|fy4|gpm|imerg|trmm|era5?|cmorph|persiann|chirps|mswep|cpc|gpcc|cru|cldas|gldas|nldas|merra|jra55|cfsr|camels|gsim|grdc)', re.IGNORECASE),
    re.compile(r'(\d{4})[/_](\d{2})'),
    re.compile(r'(\d{8})'),
    re.compile(r'(\d{4})'),
]


def normalize_source(source_raw: str) -> str:
    """
    标准化数据源名称

    Args:
        source_raw: 原始数据源名称

    Returns:
        str: 标准化后的数据源名称（小写）
    """
    source_upper = source_raw.upper()
    return SOURCE_ALIAS_MAP.get(source_upper, source_raw.lower())


def parse_filename(filename: Union[str, Path]) -> ParsedFileInfo:
    """
    从文件名中提取时间和数据源信息

    Args:
        filename: 文件名或文件路径

    Returns:
        ParsedFileInfo: 解析后的文件信息

    Examples:
        >>> info = parse_filename("FY3G_20240701_03.csv")
        >>> info.source
        'fy3g'
        >>> info.datetime
        datetime(2024, 7, 1, 3, 0)

        >>> info = parse_filename("20240701_08_GPM.csv")
        >>> info.source
        'gpm'
    """
    if isinstance(filename, Path):
        filename = filename.name

    result = ParsedFileInfo(original_name=filename)

    for pattern, pattern_type in FILENAME_PATTERNS:
        match = pattern.match(filename)
        if match:
            groups = match.groups()

            if pattern_type == 'source_date_time':
                result.source = normalize_source(groups[0])
                result.date = groups[1]
                result.time = groups[2]
                result.datetime = datetime.strptime(f"{groups[1]}{groups[2]}", "%Y%m%d%H")

            elif pattern_type == 'source_date_hhmm':
                result.source = normalize_source(groups[0])
                result.date = groups[1]
                result.time = groups[2][:2]
                result.datetime = datetime.strptime(f"{groups[1]}{groups[2]}", "%Y%m%d%H%M")

            elif pattern_type == 'source_date':
                result.source = normalize_source(groups[0])
                result.date = groups[1]
                result.datetime = datetime.strptime(groups[1], "%Y%m%d")

            elif pattern_type == 'date_time_source':
                result.date = groups[0]
                result.time = groups[1]
                result.source = normalize_source(groups[2])
                result.datetime = datetime.strptime(f"{groups[0]}{groups[1]}", "%Y%m%d%H")

            elif pattern_type == 'date_source':
                result.date = groups[0]
                result.source = normalize_source(groups[1])
                result.datetime = datetime.strptime(groups[0], "%Y%m%d")

            elif pattern_type == 'ymd_time_source':
                date_str = f"{groups[0]}{groups[1]}{groups[2]}"
                result.date = date_str
                result.time = groups[3]
                result.source = normalize_source(groups[4])
                result.datetime = datetime.strptime(f"{date_str}{groups[3]}", "%Y%m%d%H")

            elif pattern_type == 'source_ymd_time':
                result.source = normalize_source(groups[0])
                date_str = f"{groups[1]}{groups[2]}{groups[3]}"
                result.date = date_str
                result.time = groups[4]
                result.datetime = datetime.strptime(f"{date_str}{groups[4]}", "%Y%m%d%H")

            elif pattern_type == 'yyyymmdd_time_source':
                date_str = f"{groups[0]}{groups[1]}{groups[2]}"
                result.date = date_str
                result.time = groups[3]
                result.source = normalize_source(groups[4])
                result.datetime = datetime.strptime(f"{date_str}{groups[3]}", "%Y%m%d%H")

            elif pattern_type == 'source_yyyymmdd_time':
                result.source = normalize_source(groups[0])
                date_str = f"{groups[1]}{groups[2]}{groups[3]}"
                result.date = date_str
                result.time = groups[4]
                result.datetime = datetime.strptime(f"{date_str}{groups[4]}", "%Y%m%d%H")

            return result

    return result


def parse_folder(folder: Union[str, Path]) -> dict:
    """
    从文件夹路径中提取信息

    Args:
        folder: 文件夹路径

    Returns:
        dict: 包含提取信息的字典
            - source: 数据源（如果识别到）
            - year: 年份（如果识别到）
            - month: 月份（如果识别到）
            - date: 日期（如果识别到）

    Examples:
        >>> parse_folder("data/fy3g/2024/07")
        {'source': 'fy3g', 'year': '2024', 'month': '07'}

        >>> parse_folder("gpm/csv_h")
        {'source': 'gpm'}
    """
    if isinstance(folder, str):
        folder = Path(folder)

    result = {}

    path_str = str(folder).replace('\\', '/')

    for pattern in FOLDER_PATTERNS:
        match = pattern.search(path_str)
        if match:
            groups = match.groups()

            if len(groups) == 1:
                group = groups[0].lower()
                if group.isdigit():
                    if len(group) == 8:
                        result['date'] = group
                    elif len(group) == 4:
                        result['year'] = group
                else:
                    if 'source' not in result:
                        result['source'] = normalize_source(group)

            elif len(groups) == 2:
                if groups[0].isdigit() and groups[1].isdigit():
                    result['year'] = groups[0]
                    result['month'] = groups[1]

    return result


def parse_path(path: Union[str, Path]) -> ParsedFileInfo:
    """
    从完整路径（文件+文件夹）中提取信息

    优先使用文件名中的信息，文件夹信息作为补充。

    Args:
        path: 文件完整路径

    Returns:
        ParsedFileInfo: 解析后的文件信息

    Examples:
        >>> info = parse_path("data/fy3g/2024/07/FY3G_20240701_03.csv")
        >>> info.source
        'fy3g'
        >>> info.datetime
        datetime(2024, 7, 1, 3, 0)
    """
    if isinstance(path, str):
        path = Path(path)

    result = parse_filename(path.name)

    folder_info = parse_folder(path.parent)

    if result.source is None and 'source' in folder_info:
        result.source = folder_info['source']

    return result


def batch_parse_files(folder: Union[str, Path], 
                      pattern: str = "*.csv",
                      recursive: bool = False) -> list[ParsedFileInfo]:
    """
    批量解析文件夹中的文件

    Args:
        folder: 文件夹路径
        pattern: 文件匹配模式，默认 "*.csv"
        recursive: 是否递归搜索子文件夹

    Returns:
        list[ParsedFileInfo]: 解析后的文件信息列表

    Examples:
        >>> files = batch_parse_files("data/fy3g/csv_h")
        >>> len(files)
        300
        >>> files[0].source
        'fy3g'
    """
    if isinstance(folder, str):
        folder = Path(folder)

    if recursive:
        files = list(folder.rglob(pattern))
    else:
        files = list(folder.glob(pattern))

    results = []
    for f in files:
        info = parse_path(f)
        results.append(info)

    return results


def get_datetime_range(files: list[ParsedFileInfo]) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    获取文件列表的时间范围

    Args:
        files: ParsedFileInfo 列表

    Returns:
        tuple: (开始时间, 结束时间)
    """
    datetimes = [f.datetime for f in files if f.datetime is not None]
    if not datetimes:
        return None, None
    return min(datetimes), max(datetimes)


def get_unique_sources(files: list[ParsedFileInfo]) -> list[str]:
    """
    获取文件列表中的唯一数据源

    Args:
        files: ParsedFileInfo 列表

    Returns:
        list: 数据源列表
    """
    sources = set(f.source for f in files if f.source is not None)
    return sorted(sources)
