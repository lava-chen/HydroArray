"""
HydroArray CLI - 命令行接口

Usage:
    hydro --help
    hydro calc runoff --precipitation <file> --soil-moisture <file> [--output json]
    hydro metrics nse --observed <file> --simulated <file>
    hydro io read-table --path <file>
"""

from .main import main

__all__ = ["main"]