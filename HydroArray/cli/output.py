"""统一输出格式化模块"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class Result:
    """统一结果数据类"""
    status: str
    data: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    message: Optional[str] = None
    details: Optional[Dict] = None
    meta: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """转换为字典，自动跳过 None 值"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False,
                         default=_json_serializer)

    def to_compact(self) -> str:
        """转换为紧凑 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False,
                        separators=(",", ":"), default=_json_serializer)


class OutputFormatter:
    """统一的输出格式化器"""

    @staticmethod
    def format(data: Any, mode: str = "json", indent: int = 2, **kwargs) -> str:
        """格式化输出

        Args:
            data: 要输出的数据
            mode: 输出格式 (json, text, compact)
            indent: JSON 缩进
            **kwargs: 额外参数

        Returns:
            格式化后的字符串
        """
        if mode == "json":
            return OutputFormatter.to_json(data, indent=indent, **kwargs)
        elif mode == "compact":
            return OutputFormatter.to_compact(data)
        else:
            return OutputFormatter.to_text(data)

    @staticmethod
    def to_json(data: Any, indent: int = 2, **kwargs) -> str:
        """转换为 JSON 格式"""
        if isinstance(data, Result):
            return data.to_json(indent=indent)
        return json.dumps(data, indent=indent, ensure_ascii=False,
                         default=_json_serializer, **kwargs)

    @staticmethod
    def to_compact(data: Any) -> str:
        """转换为紧凑 JSON 格式（无缩进）"""
        if isinstance(data, Result):
            return data.to_compact()
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"),
                        default=_json_serializer)

    @staticmethod
    def to_text(data: Any, indent: int = 2) -> str:
        """转换为可读文本格式"""
        if isinstance(data, Result):
            data = data.to_dict()

        if isinstance(data, dict):
            return _dict_to_text(data, indent)
        elif hasattr(data, "__dict__"):
            return json.dumps(data.__dict__, indent=indent, default=_json_serializer)
        else:
            return str(data)

    @staticmethod
    def to_table(headers: List[str], rows: List[List[Any]],
                 format: str = "text") -> str:
        """转换为表格格式"""
        if format == "json":
            return json.dumps([dict(zip(headers, row)) for row in rows], indent=2)
        else:
            return _format_as_table(headers, rows)


class ProgressFormatter:
    """进度格式化器"""

    @staticmethod
    def write(message: str, progress: Optional[float] = None,
             iteration: Optional[int] = None,
             metric: Optional[Dict] = None,
             stream = None) -> None:
        """格式化并写入进度信息

        Args:
            message: 进度消息
            progress: 进度百分比 (0-1)
            iteration: 当前迭代次数
            metric: 当前最佳指标 {'name': 'NSE', 'value': 0.85}
            stream: 输出流，默认 stderr
        """
        if stream is None:
            stream = sys.stderr

        data = {"message": message}
        if progress is not None:
            data["progress"] = round(progress * 100, 1)  # 转为百分比
        if iteration is not None:
            data["iteration"] = iteration
        if metric:
            data["metric"] = metric

        stream.write(json.dumps(data) + "\n")
        stream.flush()

    @staticmethod
    def clear(stream = None) -> None:
        """清除进度行"""
        if stream is None:
            stream = sys.stderr
        stream.write("\r\033[K")
        stream.flush()


def _dict_to_text(data: Dict, indent: int = 0, prefix: str = "") -> str:
    """将字典转换为可读的文本格式"""
    lines = []
    pad = " " * indent

    for key, value in data.items():
        display_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict) and value:
            lines.append(f"{pad}{display_key}:")
            lines.append(_dict_to_text(value, indent + 2))
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                lines.append(f"{pad}{display_key}: []")
            elif len(value) <= 5:
                lines.append(f"{pad}{display_key}: {value}")
            else:
                lines.append(f"{pad}{display_key}: [{len(value)} items]")
                for i, item in enumerate(value[:3]):
                    lines.append(f"{pad}  [{i}]: {item}")
                if len(value) > 3:
                    lines.append(f"{pad}  ... ({len(value) - 3} more)")
        elif isinstance(value, float):
            lines.append(f"{pad}{display_key}: {value:.6g}")
        else:
            lines.append(f"{pad}{display_key}: {value}")

    return "\n".join(lines)


def _json_serializer(obj: Any) -> Any:
    """JSON 序列化器，处理 numpy 和 pandas 类型"""
    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        # 避免科学计数法过小
        if abs(val) < 1e-10 and val != 0:
            return f"{val:.6e}"
        return val
    elif isinstance(obj, np.ndarray):
        if obj.size > 1000:
            return {"_type": "ndarray", "shape": list(obj.shape),
                   "data": obj.flatten()[:100].tolist(), "_truncated": int(obj.size)}
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return {"_type": "DataFrame", "rows": len(obj),
                "columns": list(df.columns) if (df := obj) else [],
                "data": obj.head(5).to_dict(orient="records")}
    elif isinstance(obj, pd.Series):
        return {"_type": "Series", "length": len(obj),
                "data": obj.head(5).to_dict()}
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        try:
            return list(obj)
        except TypeError:
            return str(obj)


def success_result(data: Any = None, message: Optional[str] = None,
                   meta: Optional[Dict] = None) -> Result:
    """构建成功结果"""
    return Result(
        status="success",
        data=data,
        message=message,
        meta=meta
    )


def error_result(error: str, error_type: str = "Error",
                details: Optional[Dict] = None) -> Result:
    """构建错误结果"""
    return Result(
        status="error",
        error=str(error),
        error_type=error_type,
        details=details
    )


def format_metric_result(name: str, value: Union[float, int],
                         unit: Optional[str] = None,
                         **kwargs) -> Dict:
    """格式化指标计算结果"""
    result = {
        "metric": name.upper(),
        "value": float(value) if isinstance(value, (int, float)) else value,
    }
    if unit:
        result["unit"] = unit
    result.update(kwargs)
    return result


def _format_as_table(headers: List[str], rows: List[List[Any]],
                      max_width: int = 80) -> str:
    """格式化表格"""
    if not rows:
        return ""

    # 计算每列宽度
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # 限制最大宽度
    col_widths = [min(w, max_width // len(headers)) for w in col_widths]

    # 构建表格
    lines = []

    # 表头
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # 数据行
    for row in rows:
        line = " | ".join(str(cell)[:col_widths[i]].ljust(col_widths[i])
                         for i, cell in enumerate(row))
        lines.append(line)

    return "\n".join(lines)