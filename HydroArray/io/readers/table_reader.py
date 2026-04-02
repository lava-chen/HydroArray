"""
水文观测表格数据提取模块

支持从各种格式的水文表格中提取数据，包括：
- 多列月份格式（如降雨蒸发表格）
- 逐日平均水位表格式
- 水位观测记录格式
- 标准时间序列格式

主要功能：
- 自动检测表格结构
- 智能解析不同格式的水文数据
- 输出标准化的时间序列数据

使用示例：
    >>> from HydroArray.utils.TableReader import read_hydro_table, TableType
    >>> 
    >>> # 自动检测并读取
    >>> df = read_hydro_table("data.xlsx")
    >>> 
    >>> # 指定表格类型
    >>> df = read_hydro_table("water_level.xlsx", table_type=TableType.WATER_LEVEL_RECORD)
    >>> 
    >>> # 提取降雨蒸发数据
    >>> df = read_hydro_table("rainfall.xlsx")
    >>> rainfall_df = df[['date', 'P', 'Ep']].dropna(subset=['P', 'Ep'])
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Optional, Union, Any, cast
from datetime import datetime
import re
import warnings


MONTH_MAP: dict[str, int] = {
    '一月': 1, '二月': 2, '三月': 3, '四月': 4, '五月': 5, '六月': 6,
    '七月': 7, '八月': 8, '九月': 9, '十月': 10, '十一月': 11, '十二月': 12,
    '1月': 1, '2月': 2, '3月': 3, '4月': 4, '5月': 5, '6月': 6,
    '7月': 7, '8月': 8, '9月': 9, '10月': 10, '11月': 11, '12月': 12,
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
}

SUMMARY_KEYWORDS: list[str] = [
    '总数', '平均', '最高', '最低', '日期', '年总数', '保证率', '计算', '校核', '复核'
]

HYDRO_KEYWORDS: list[str] = ['P', 'Ep', 'E', 'Q', 'W', 'H', 'T', 'R']

MONTH_KEYWORDS: list[str] = [
    '月', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    '一月', '二月', '三月', '四月', '五月', '六月',
    '七月', '八月', '九月', '十月', '十一月', '十二月'
]


class TableType:
    MULTI_COLUMN_MONTH = "multi_column_month"  # 多列月份格式（如降雨蒸发表）
    DAILY_AVERAGE = "daily_average"  # 逐日平均水位表
    WATER_LEVEL_RECORD = "water_level_record"  # 水位观测记录格式
    CROSS_SECTION = "cross_section"  # 大断面测量数据
    TIME_SERIES = "time_series"  # 标准时间序列格式
    UNKNOWN = "unknown"


def detect_table_structure(df: pd.DataFrame) -> dict[str, Any]:
    """
    自动检测表格结构

    分析表格的行列特征，识别表格类型和关键信息位置

    Args:
        df: 原始表格数据（无表头的 DataFrame）

    Returns:
        dict: 包含以下信息：
            - 'table_type': 表格类型 (TableType 枚举值)
            - 'header_row': 表头行索引 (int 或 None)
            - 'data_start_row': 数据起始行 (int)
            - 'date_cols': 日期相关列位置 (dict)
            - 'value_cols': 数值列位置 (dict)
            - 'metadata': 其他元数据 (dict)
            - 'format_type': 格式子类型 (仅多列月份格式)
    """
    result: dict[str, Any] = {
        'table_type': TableType.UNKNOWN,
        'header_row': None,
        'data_start_row': 0,
        'date_cols': {},
        'value_cols': {},
        'metadata': {}
    }

    if len(df) == 0 or len(df.columns) == 0:
        return result

    # 分析前10行来检测结构
    sample_rows = min(10, len(df))

    # 1. 检测是否为水位观测记录格式
    wl_pattern = _detect_water_level_pattern(df.head(sample_rows))
    if wl_pattern['is_match']:
        result['table_type'] = TableType.WATER_LEVEL_RECORD
        result.update(wl_pattern)
        return result

    # 2. 检测是否为大断面测量数据
    cs_pattern = _detect_cross_section_pattern(df.head(sample_rows))
    if cs_pattern['is_match']:
        result['table_type'] = TableType.CROSS_SECTION
        result.update(cs_pattern)
        return result

    # 3. 检测是否为多列月份格式
    month_pattern = _detect_multi_column_month_pattern(df.head(sample_rows))
    if month_pattern['is_match']:
        # 根据格式类型设置正确的表格类型
        if month_pattern.get('format_type') == 'daily_average':
            result['table_type'] = TableType.DAILY_AVERAGE
        else:
            result['table_type'] = TableType.MULTI_COLUMN_MONTH
        result.update(month_pattern)
        return result

    # 3. 默认为时间序列格式
    result['table_type'] = TableType.TIME_SERIES
    ts_pattern = _detect_time_series_pattern(df.head(sample_rows))
    result.update(ts_pattern)

    return result


def _detect_water_level_pattern(df: pd.DataFrame) -> dict[str, Any]:
    """
    检测水位观测记录格式

    特征：
    - 包含"日"、"时"、"分"等时间相关列
    - 包含"水尺"、"水位"等关键词
    """
    result: dict[str, Any] = {'is_match': False}

    # 检查所有行的文本内容
    all_text = df.astype(str).values.flatten()
    all_text_str = ' '.join(all_text)

    # 水位表关键词
    wl_keywords = ['日', '时', '分', '水尺', '水位', '水尺读数']
    wl_keyword_count = sum(1 for kw in wl_keywords if kw in all_text_str)

    if wl_keyword_count >= 3:
        result['is_match'] = True

        # 查找表头行
        for idx, row in df.iterrows():
            row_text = ' '.join(row.astype(str).values)
            if '日' in row_text and ('时' in row_text or '分' in row_text):
                result['header_row'] = idx
                break

        # 识别列位置
        if result.get('header_row') is not None:
            header_row = df.iloc[cast(int, result['header_row'])]
            result['date_cols'] = _identify_date_columns(header_row)
            result['value_cols'] = _identify_value_columns(header_row)
            result['data_start_row'] = cast(int, result['header_row']) + 1

    return result


def _detect_cross_section_pattern(df: pd.DataFrame) -> dict[str, Any]:
    """
    检测大断面测量数据格式

    特征：
    - 包含"点次"、"起点距"、"河底高程"等列名
    - 可能左右分栏
    """
    result: dict[str, Any] = {'is_match': False}

    all_text = df.astype(str).values.flatten()
    all_text_str = ' '.join(all_text)

    cs_keywords = ['点次', '起点距', '河底高程']
    cs_keyword_count = sum(1 for kw in cs_keywords if kw in all_text_str)

    if cs_keyword_count >= 2:
        result['is_match'] = True

        for idx, row in df.iterrows():
            row_text = ' '.join(row.astype(str).values)
            if '点次' in row_text and '起点距' in row_text:
                result['header_row'] = idx
                break

        if result.get('header_row') is not None:
            result['data_start_row'] = cast(int, result['header_row']) + 1

    return result


def _detect_multi_column_month_pattern(df: pd.DataFrame) -> dict[str, Any]:
    """
    检测多列月份格式

    特征：
    - 第一行包含月份名称
    - 可能是：
      a) 第二行包含属性标识（P、Ep等）- 降雨蒸发表格式
      b) 第一列是日期（1-31），数据是数值 - 逐日平均水位表格式
    """
    result: dict[str, Any] = {'is_match': False}

    if len(df) < 2:
        return result

    # 获取第一行和第二行
    first_row = df.iloc[0].astype(str)
    second_row = df.iloc[1].astype(str)

    has_month = any(any(kw in str(cell) for kw in MONTH_KEYWORDS)
                    for cell in first_row if pd.notna(cell))

    if not has_month:
        return result

    has_hydro_attr = any(any(str(cell).strip() == kw for kw in HYDRO_KEYWORDS)
                         for cell in second_row if pd.notna(cell))

    if has_hydro_attr:
        result['is_match'] = True
        result['format_type'] = 'rainfall_evaporation'
        result['header_rows'] = [0, 1]
        result['data_start_row'] = 2
    else:
        first_cell = str(first_row.iloc[0]).strip() if len(first_row) > 0 else ''
        date_header_keywords = ['月日', '日期', '日', 'day', 'Day', 'DAY', 'Date', 'DATE']
        has_date_header = any(kw in first_cell for kw in date_header_keywords)

        if has_date_header:
            result['is_match'] = True
            result['format_type'] = 'daily_average'
            result['header_row'] = 0
            result['data_start_row'] = 1
        else:
            first_col = df.iloc[1:, 0].astype(str)
            numeric_count = sum(1 for v in first_col if str(v).strip().isdigit())
            total_count = len(first_col)
            if numeric_count >= 8 or (total_count > 0 and numeric_count / total_count > 0.8):
                result['is_match'] = True
                result['format_type'] = 'daily_average'
                result['header_row'] = 0
                result['data_start_row'] = 1
                warnings.warn(
                    "检测到可能的逐日平均水位表格式，但第一列第一行不是日期标识。"
                    "建议检查表格格式是否正确。",
                    UserWarning
                )

    return result


def _detect_time_series_pattern(df: pd.DataFrame) -> dict[str, Any]:
    """
    检测标准时间序列格式

    特征：
    - 第一行或前几行包含日期、时间等列名
    """
    result: dict[str, Any] = {
        'is_match': True,
        'header_row': 0,
        'data_start_row': 1,
        'date_cols': {},
        'value_cols': {}
    }

    header_row = df.iloc[0]
    result['date_cols'] = _identify_date_columns(header_row)
    result['value_cols'] = _identify_value_columns(header_row)

    return result


def _identify_date_columns(header_row: pd.Series) -> dict:
    """
    识别日期相关列
    """
    date_cols = {}

    date_keywords = {
        'year': ['年', 'year', 'Year', 'YEAR'],
        'month': ['月', 'month', 'Month', 'MONTH', 'M'],
        'day': ['日', 'day', 'Day', 'DAY', 'D', '日期', 'date', 'Date', 'DATE'],
        'hour': ['时', 'hour', 'Hour', 'HOUR', 'H', '时间', 'time', 'Time'],
        'minute': ['分', 'minute', 'Minute', 'MIN', 'Min', 'min'],
    }

    for idx, value in header_row.items():
        if pd.isna(value):
            continue
        value_str = str(value).strip()

        for col_type, keywords in date_keywords.items():
            if any(kw in value_str for kw in keywords):
                date_cols[col_type] = idx
                break

    return date_cols


def _identify_value_columns(header_row: pd.Series) -> dict:
    """
    识别数值列
    """
    value_cols = {}

    value_keywords = {
        'water_level': ['水位', '水位(m)', 'water_level', 'WaterLevel', 'Z'],
        'water_level_reading': ['水尺读数', '读数', 'reading', 'Reading'],
        'rainfall': ['降雨', '雨量', 'P', 'p', '降雨量', 'rainfall', 'Rainfall'],
        'evaporation': ['蒸发', '蒸发能力', 'Ep', 'EP', 'evaporation', 'Evaporation'],
        'discharge': ['流量', 'Q', 'discharge', 'Discharge'],
        'daily_avg': ['日平均', '日均', 'daily', 'Daily', '平均'],
    }

    for idx, value in header_row.items():
        if pd.isna(value):
            continue
        value_str = str(value).strip()

        for col_type, keywords in value_keywords.items():
            if any(kw in value_str for kw in keywords):
                if col_type not in value_cols:
                    value_cols[col_type] = []
                value_cols[col_type].append(idx)
                break

    return value_cols


def read_hydro_table(
    file_path: str | Path,
    sheet_name: str | int = 0,
    table_type: str | None = None,
    auto_detect: bool = True,
) -> pd.DataFrame:
    """
    智能读取水文观测表格数据

    自动检测表格结构并提取数据

    Args:
        file_path: 表格文件路径（支持 .xlsx, .xls, .csv）
        sheet_name: 工作表名称或索引，默认为 0
        table_type: 指定表格类型，为 None 时自动检测。
            可选值：TableType.WATER_LEVEL_RECORD, TableType.DAILY_AVERAGE,
            TableType.MULTI_COLUMN_MONTH, TableType.TIME_SERIES
        auto_detect: 是否自动检测表格结构，默认为 True

    Returns:
        pd.DataFrame: 标准化后的时间序列数据，包含以下可能的列：
            - date/datetime: 日期时间
            - year, month, day, hour, minute: 时间分量
            - P: 降雨量
            - Ep: 蒸发能力
            - water_level: 水位
            - water_level_reading: 水尺读数
            - source: 数据来源文件名

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式或表格类型无效
        RuntimeError: 数据解析失败

    Examples:
        >>> # 自动检测并读取
        >>> df = read_hydro_table("data.xlsx")
        >>>
        >>> # 指定表格类型
        >>> df = read_hydro_table("data.xlsx", table_type=TableType.WATER_LEVEL_RECORD)
        >>>
        >>> # 提取降雨蒸发数据
        >>> df = read_hydro_table("rainfall.xlsx")
        >>> rainfall_df = df[['date', 'P', 'Ep']].dropna(subset=['P', 'Ep'])
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in ['.xlsx', '.xls']:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    elif suffix == '.csv':
        df_raw = pd.read_csv(file_path, header=None)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，仅支持 .xlsx, .xls, .csv")

    if df_raw.empty:
        warnings.warn(f"文件 {file_path.name} 为空，返回空 DataFrame", UserWarning)
        return pd.DataFrame()

    if len(df_raw.columns) == 0:
        warnings.warn(f"文件 {file_path.name} 没有列，返回空 DataFrame", UserWarning)
        return pd.DataFrame()

    if auto_detect and table_type is None:
        structure = detect_table_structure(df_raw)
        table_type = structure['table_type']
    else:
        structure = detect_table_structure(df_raw)

    if table_type == TableType.UNKNOWN:
        warnings.warn(
            f"无法识别表格类型，将尝试作为标准时间序列格式解析。"
            f"如果结果不正确，请手动指定 table_type 参数。",
            UserWarning
        )
        table_type = TableType.TIME_SERIES

    try:
        if table_type == TableType.WATER_LEVEL_RECORD:
            result = _parse_water_level_record(df_raw, structure, file_path.name)
        elif table_type == TableType.DAILY_AVERAGE:
            result = _parse_daily_average_table(df_raw, structure, file_path.name)
        elif table_type == TableType.MULTI_COLUMN_MONTH:
            format_type = structure.get('format_type', 'rainfall_evaporation')
            if format_type == 'daily_average':
                result = _parse_daily_average_table(df_raw, structure, file_path.name)
            else:
                result = _parse_rainfall_evaporation_table(df_raw, structure, file_path.name)
        elif table_type == TableType.CROSS_SECTION:
            result = _parse_cross_section_table(df_raw, structure, file_path.name)
        elif table_type == TableType.TIME_SERIES:
            result = _parse_time_series(df_raw, structure, file_path.name)
        else:
            raise ValueError(f"无效的表格类型: {table_type}")
    except Exception as e:
        raise RuntimeError(f"解析表格 {file_path.name} 失败: {str(e)}") from e

    if result.empty:
        warnings.warn(
            f"从文件 {file_path.name} 解析出的数据为空，请检查表格格式是否正确。",
            UserWarning
        )

    return result


def _parse_water_level_record(
    df: pd.DataFrame,
    structure: dict,
    source_name: str = "unknown"
) -> pd.DataFrame:
    """
    解析水位观测记录格式

    支持左右分栏的表格格式
    """
    records = []

    # 获取表头行
    header_row_idx = structure.get('header_row', 0)
    data_start_row = structure.get('data_start_row', 1)

    # 获取表头内容
    header_row = df.iloc[header_row_idx]

    # 识别所有日期列组（支持左右分栏）
    # 查找所有包含"日"、"时"、"分"的列位置
    day_cols = []
    hour_cols = []
    minute_cols = []
    wl_reading_cols = []
    water_level_cols = []

    for idx, value in enumerate(header_row):
        if pd.isna(value):
            continue
        value_str = str(value).strip()

        if value_str == '日' or value_str in ['day', 'Day', 'DAY', '日期']:
            day_cols.append(idx)
        elif value_str == '时' or value_str in ['hour', 'Hour', 'HOUR', 'H']:
            hour_cols.append(idx)
        elif value_str == '分' or value_str in ['minute', 'Minute', 'MIN']:
            minute_cols.append(idx)
        elif '水尺读数' in value_str or 'reading' in value_str.lower():
            wl_reading_cols.append(idx)
        elif value_str == '水位(m)' or '水位' in value_str:
            water_level_cols.append(idx)

    # 如果没有识别到时/分列，使用默认值
    if not hour_cols and day_cols:
        hour_cols = [d + 1 for d in day_cols]
    if not minute_cols and day_cols:
        minute_cols = [d + 2 for d in day_cols]

    # 获取数据部分
    data_df = df.iloc[data_start_row:].copy()

    # 对每个分栏的日列进行向前填充（处理合并单元格）
    for day_col in day_cols:
        if day_col < len(data_df.columns):
            data_df.iloc[:, day_col] = data_df.iloc[:, day_col].ffill()

    # 默认使用当前年月
    year = datetime.now().year
    month = datetime.now().month

    # 处理每个分栏
    for col_group_idx in range(len(day_cols)):
        day_col = day_cols[col_group_idx] if col_group_idx < len(day_cols) else day_cols[0]
        hour_col = hour_cols[col_group_idx] if col_group_idx < len(hour_cols) else hour_cols[0]
        minute_col = minute_cols[col_group_idx] if col_group_idx < len(minute_cols) else minute_cols[0]

        # 找到对应的水尺读数和水位列
        # 通常在当前日期组之后
        wl_reading_col = None
        water_level_col = None

        for wl_col in wl_reading_cols:
            if day_col < wl_col < (day_cols[col_group_idx + 1] if col_group_idx + 1 < len(day_cols) else len(header_row)):
                wl_reading_col = wl_col
                break

        for wl_col in water_level_cols:
            if day_col < wl_col < (day_cols[col_group_idx + 1] if col_group_idx + 1 < len(day_cols) else len(header_row)):
                water_level_col = wl_col
                break

        for _, row in data_df.iterrows():
            try:
                day = _to_numeric(row.iloc[day_col])
                hour = _to_numeric(row.iloc[hour_col]) if hour_col < len(row) else 0
                minute = _to_numeric(row.iloc[minute_col]) if minute_col < len(row) else 0

                if pd.isna(day) or day <= 0 or day > 31:
                    continue

                # 构建日期时间
                dt = pd.Timestamp(year=year, month=month, day=int(day),
                                hour=int(hour), minute=int(minute))

                record = {
                    'datetime': dt,
                    'date': dt.date(),
                    'year': year,
                    'month': month,
                    'day': int(day),
                    'hour': int(hour),
                    'minute': int(minute),
                    'source': source_name,
                }

                # 提取水尺读数
                if wl_reading_col is not None and wl_reading_col < len(row):
                    val = _to_numeric(row.iloc[wl_reading_col])
                    if not pd.isna(val):
                        record['water_level_reading'] = val

                # 提取水位
                if water_level_col is not None and water_level_col < len(row):
                    val = _to_numeric(row.iloc[water_level_col])
                    if not pd.isna(val):
                        record['water_level'] = val

                # 只添加有数值数据的记录
                if 'water_level_reading' in record or 'water_level' in record:
                    records.append(record)

            except (ValueError, TypeError):
                continue

    result_df = pd.DataFrame(records)

    # 按时间排序
    if not result_df.empty and 'datetime' in result_df.columns:
        result_df = result_df.sort_values('datetime').reset_index(drop=True)

    return result_df


def _parse_cross_section_table(
    df: pd.DataFrame,
    structure: dict,
    source_name: str = "unknown"
) -> pd.DataFrame:
    """
    解析大断面测量数据格式

    支持左右分栏的表格格式
    """
    records = []

    header_row_idx = structure.get('header_row', 0)
    data_start_row = structure.get('data_start_row', 1)

    header_row = df.iloc[header_row_idx]

    point_cols = []
    distance_cols = []
    elevation_cols = []

    for idx, value in enumerate(header_row):
        if pd.isna(value):
            continue
        value_str = str(value).strip()

        if value_str == '点次':
            point_cols.append(idx)
        elif '起点距' in value_str:
            distance_cols.append(idx)
        elif '河底高程' in value_str:
            elevation_cols.append(idx)

    data_df = df.iloc[data_start_row:].copy()

    for col_group_idx in range(len(point_cols)):
        point_col = point_cols[col_group_idx] if col_group_idx < len(point_cols) else None
        distance_col = distance_cols[col_group_idx] if col_group_idx < len(distance_cols) else None
        elevation_col = elevation_cols[col_group_idx] if col_group_idx < len(elevation_cols) else None

        if distance_col is None or elevation_col is None:
            continue

        for _, row in data_df.iterrows():
            try:
                distance = _to_numeric(row.iloc[distance_col]) if distance_col < len(row) else None
                elevation = _to_numeric(row.iloc[elevation_col]) if elevation_col < len(row) else None

                if pd.isna(distance) or pd.isna(elevation):
                    continue

                record = {
                    'source': source_name,
                    'distance': distance,
                    'elevation': elevation,
                }

                if point_col is not None and point_col < len(row):
                    point_val = row.iloc[point_col]
                    if pd.notna(point_val):
                        record['point'] = str(point_val).strip()

                records.append(record)

            except (ValueError, TypeError):
                continue

    result_df = pd.DataFrame(records)

    if not result_df.empty:
        result_df = result_df.sort_values('distance').reset_index(drop=True)

    return result_df


def _parse_rainfall_evaporation_table(
    df: pd.DataFrame,
    structure: dict,
    source_name: str = "unknown"
) -> pd.DataFrame:
    """
    解析降雨蒸发表格式
    """
    records = []

    header_rows = structure.get('header_rows', [0, 1])
    data_start_row = structure.get('data_start_row', 2)

    header_row_1 = df.iloc[header_rows[0]]
    header_row_2 = df.iloc[header_rows[1]]

    data_df = df.iloc[data_start_row:].copy()

    date_col = 0

    column_info = []
    current_month = None
    for i in range(1, len(header_row_1)):
        month_cell = header_row_1.iloc[i]
        attr = header_row_2.iloc[i] if pd.notna(header_row_2.iloc[i]) else None

        if pd.notna(month_cell) and str(month_cell).strip() and str(month_cell).strip().lower() != 'nan':
            current_month = str(month_cell).replace(' ', '').replace('\n', '')

        if current_month and attr:
            attr_clean = str(attr).strip()
            column_info.append({
                'col_idx': i,
                'month': current_month,
                'attribute': attr_clean,
            })

    months = set(info['month'] for info in column_info)

    year = datetime.now().year

    date_records = {}

    for _, row in data_df.iterrows():
        day = _to_numeric(row.iloc[date_col])
        if pd.isna(day):
            continue

        for month_name in months:
            month_num = MONTH_MAP.get(month_name, 6)

            try:
                dt = pd.Timestamp(year=year, month=month_num, day=int(day))
            except ValueError:
                continue

            key = (year, month_num, int(day))
            if key not in date_records:
                date_records[key] = {
                    'date': dt,
                    'year': year,
                    'month': month_num,
                    'day': int(day),
                    'source': source_name,
                }

            for info in column_info:
                if info['month'] == month_name:
                    attr = info['attribute']
                    col_idx = info['col_idx']
                    if col_idx < len(row):
                        val = _to_numeric(row.iloc[col_idx])
                        if not pd.isna(val):
                            date_records[key][attr] = val

    records = list(date_records.values())
    result_df = pd.DataFrame(records)

    if not result_df.empty and 'date' in result_df.columns:
        result_df = result_df.sort_values('date').reset_index(drop=True)

    return result_df


def _parse_daily_average_table(
    df: pd.DataFrame,
    structure: dict,
    source_name: str = "unknown"
) -> pd.DataFrame:
    """
    解析逐日平均水位表格式

    特征：
    - 第一行是月份名称
    - 第一列是日期（1-31）
    - 数据是逐日平均水位值
    - 后面可能有总结统计行需要过滤
    """
    records = []

    header_row = structure.get('header_row', 0)
    data_start_row = structure.get('data_start_row', 1)

    # 获取月份行
    header_row_data = df.iloc[header_row].astype(str)

    # 数据从指定行开始
    data_df = df.iloc[data_start_row:].copy()

    date_col = 0

    month_columns = []
    for i in range(1, len(header_row_data)):
        cell_value = str(header_row_data.iloc[i]).strip()
        for month_name, month_num in MONTH_MAP.items():
            if month_name in cell_value:
                month_columns.append({
                    'col_idx': i,
                    'month_name': month_name,
                    'month_num': month_num
                })
                break

    year = datetime.now().year

    date_records = {}

    for _, row in data_df.iterrows():
        first_cell = str(row.iloc[date_col]).strip()
        if any(kw in first_cell for kw in SUMMARY_KEYWORDS):
            continue

        # 尝试解析日期
        try:
            day = int(float(first_cell))
            if day < 1 or day > 31:
                continue
        except (ValueError, TypeError):
            continue

        # 提取各月份的水位数据
        for month_info in month_columns:
            month_num = month_info['month_num']
            col_idx = month_info['col_idx']

            try:
                dt = pd.Timestamp(year=year, month=month_num, day=day)
            except ValueError:
                # 无效日期（如2月30日）
                continue

            # 提取水位值
            if col_idx < len(row):
                raw_cell = row.iloc[col_idx]
                val = _to_numeric(raw_cell)
                if not pd.isna(val):
                    key = (year, month_num, day)
                    # 判断是否为完整水位值（字符串包含小数点）
                    is_full_value = isinstance(raw_cell, str) and '.' in raw_cell

                    if key not in date_records:
                        date_records[key] = {
                            'date': dt,
                            'year': year,
                            'month': month_num,
                            'day': day,
                            'source': source_name,
                            'water_level_raw': val,
                            'is_full_value': is_full_value,
                        }
                    else:
                        date_records[key]['water_level_raw'] = val
                        date_records[key]['is_full_value'] = is_full_value

    # 处理水位值的特殊格式（省略整数部分）
    # 按月份分组处理
    for month_num in set(k[1] for k in date_records.keys()):
        month_keys = [k for k in date_records.keys() if k[1] == month_num]
        month_keys.sort()  # 按日期排序

        # 找到该月的第一个完整水位值
        base_int = None
        for key in month_keys:
            if date_records[key]['is_full_value']:
                base_int = int(date_records[key]['water_level_raw'])
                break

        # 如果没有找到完整值，使用第一个值作为基准
        if base_int is None and month_keys:
            base_int = int(date_records[month_keys[0]]['water_level_raw'])

        # 处理每个值
        for key in month_keys:
            raw_val = date_records[key]['water_level_raw']
            is_full = date_records[key]['is_full_value']

            if is_full:
                # 完整水位值
                water_level = raw_val
                base_int = int(raw_val)  # 更新整数部分基准
            else:
                # 省略整数部分的值，格式为 XX.YY 中的 YY 部分
                # 例如：87 表示 34.87，18 表示 35.18，04 表示 35.04
                water_level = base_int + raw_val / 100

            date_records[key]['water_level'] = water_level
            # 删除临时字段
            del date_records[key]['water_level_raw']
            del date_records[key]['is_full_value']

    records = list(date_records.values())
    result_df = pd.DataFrame(records)

    if not result_df.empty and 'date' in result_df.columns:
        result_df = result_df.sort_values('date').reset_index(drop=True)

    return result_df


def _parse_time_series(
    df: pd.DataFrame,
    structure: dict,
    source_name: str = "unknown"
) -> pd.DataFrame:
    """
    解析标准时间序列格式
    """
    header_row = structure.get('header_row', 0)
    data_start_row = structure.get('data_start_row', 1)

    # 使用第一行作为列名
    columns = df.iloc[header_row].astype(str).tolist()
    data_df = df.iloc[data_start_row:].copy()
    data_df.columns = columns

    # 尝试识别日期列
    date_col = None
    for col in columns:
        if any(kw in str(col) for kw in ['date', 'Date', 'DATE', '日期', '时间', 'time', 'Time']):
            date_col = col
            break

    if date_col:
        data_df[date_col] = pd.to_datetime(data_df[date_col], errors='coerce')
        data_df = data_df.dropna(subset=[date_col])

    data_df['source'] = source_name

    return data_df.reset_index(drop=True)


def _to_numeric(value) -> float:
    """
    将值转换为数值，失败返回 NaN

    Args:
        value: 待转换的值

    Returns:
        float: 转换后的数值，失败返回 np.nan
    """
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def extract_time_series(
    df: pd.DataFrame,
    date_col: str = 'date',
    value_col: str | list[str] = 'P',
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    从解析后的数据中提取时间序列

    Args:
        df: read_hydro_table 返回的 DataFrame
        date_col: 日期列名，默认为 'date'
        value_col: 数值列名或列表，默认为 'P'
        drop_na: 是否删除缺失值，默认为 True

    Returns:
        pd.DataFrame: 仅包含指定列的时间序列数据

    Examples:
        >>> df = read_hydro_table("rainfall.xlsx")
        >>> # 提取单列
        >>> rainfall = extract_time_series(df, value_col='P')
        >>> # 提取多列
        >>> multi = extract_time_series(df, value_col=['P', 'Ep'])
    """
    if isinstance(value_col, str):
        cols = [date_col, value_col]
    else:
        cols = [date_col] + list(value_col)

    # 只保留存在的列
    cols = [c for c in cols if c in df.columns]
    result = df[cols].copy()

    if drop_na:
        result = result.dropna()

    return result

