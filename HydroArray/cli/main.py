"""HydroArray CLI 入口点"""

from __future__ import annotations

import argparse
import os
import sys
import json
import signal
from typing import Optional, Callable

from .output import OutputFormatter, error_result, success_result
from . import calc, model, metrics, io, update


def _get_output_format() -> str:
    """获取输出格式，优先级：命令行参数 > 环境变量 > 默认值"""
    # 检查环境变量
    env_output = os.environ.get("HYDRO_OUTPUT", "").lower()
    if env_output in ("json", "text", "compact"):
        return env_output

    # 检查命令行参数（在 parse_known_args 阶段）
    for arg in sys.argv:
        if arg == "--json":
            return "json"
        elif arg == "--text":
            return "text"
        elif arg == "--compact":
            return "compact"

    return "json"


def _register_global_flags(parser: argparse.ArgumentParser) -> None:
    """注册全局参数"""
    parser.add_argument(
        "--version",
        action="version",
        version=f"HydroArray CLI {__version__()}"
    )

    # 输出格式
    parser.add_argument(
        "--output", "-o",
        choices=["json", "text", "compact"],
        default=None,  # 稍后从环境变量或默认值填充
        help="输出格式 (默认: json，可通过 HYDRO_OUTPUT 环境变量设置)"
    )

    # 输出格式简写
    parser.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="output",
        help="输出 JSON 格式 (等同于 --output json)"
    )
    parser.add_argument(
        "--text",
        action="store_const",
        const="text",
        dest="output",
        help="输出文本格式 (等同于 --output text)"
    )
    parser.add_argument(
        "--compact",
        action="store_const",
        const="compact",
        dest="output",
        help="输出紧凑 JSON 格式 (等同于 --output compact)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出到 stderr"
    )

    # 配置覆盖参数
    parser.add_argument(
        "--config", "-c",
        help="JSON 格式的配置参数，覆盖默认值 (格式: '{\"key\": \"value\"}')"
    )


def _parse_known_args() -> tuple:
    """先解析全局参数，再获取输出格式"""
    parser = argparse.ArgumentParser(add_help=False)
    _register_global_flags(parser)
    return parser.parse_known_args()


class ProgressWriter:
    """进度输出器，写入 stderr"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_line_len = 0

    def write(self, message: str, progress: Optional[float] = None,
              iteration: Optional[int] = None, **kwargs) -> None:
        """写入进度信息到 stderr"""
        if not self.enabled:
            return

        data = {"message": message}
        if progress is not None:
            data["progress"] = round(progress, 3)
        if iteration is not None:
            data["iteration"] = iteration
        data.update(kwargs)

        # 清除上一行
        if self._last_line_len > 0:
            sys.stderr.write("\r" + " " * self._last_line_len + "\r")
            self._last_line_len = 0

        output = json.dumps(data)
        sys.stderr.write(output)
        self._last_line_len = len(output)

    def flush(self) -> None:
        """刷新 stderr"""
        sys.stderr.flush()

    def clear(self) -> None:
        """清除进度行"""
        if self._last_line_len > 0:
            sys.stderr.write("\r" + " " * self._last_line_len + "\r")
            self._last_line_len = 0


def main():
    """主入口函数"""
    # 解析已知参数，获取全局设置
    known_args, remaining = _parse_known_args()

    # 确定输出格式
    output_format = (known_args.output if known_args.output
                     else os.environ.get("HYDRO_OUTPUT", "json").lower())

    # 解析配置
    config_overrides = {}
    if known_args.config:
        try:
            config_overrides = json.loads(known_args.config)
        except json.JSONDecodeError as e:
            print(json.dumps(error_result(
                f"配置 JSON 格式错误: {e}",
                "ConfigError"
            )), file=sys.stderr)
            sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="hydro",
        description="HydroArray CLI - 水文数据处理命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
环境变量:
  HYDRO_OUTPUT=json|text|compact  设置默认输出格式

示例:
  hydro calc runoff --precipitation data/precip.csv --soil-moisture data/soil.csv
  hydro metrics nse --observed obs.csv --simulated sim.csv
  hydro model list
  hydro io read-table --path data.csv --datetime timestamp
  hydro update check
  HYDRO_OUTPUT=text hydro model list  # 使用环境变量设置输出格式

更多信息: https://github.com/lava-chen/HydroArray
        """
    )

    # 添加全局参数
    _register_global_flags(parser)

    # 添加子命令
    subparsers = parser.add_subparsers(
        dest="command",
        title="命令",
        description="可用命令列表"
    )

    # 注册各模块的子命令
    calc.register(subparsers)
    model.register(subparsers)
    metrics.register(subparsers)
    io.register(subparsers)
    update.register(subparsers)

    args = parser.parse_args()

    # 没有子命令时显示帮助
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # 应用配置覆盖
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # 创建进度写入器
    progress = ProgressWriter(enabled=known_args.verbose)

    # 信号处理（支持 Ctrl+C）
    def handle_signal(signum, frame):
        print(json.dumps(error_result("操作被用户中断", "KeyboardInterrupt")),
              file=sys.stderr)
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, handle_signal)

    # 执行命令
    try:
        if hasattr(args, 'func'):
            # 传递进度写入器和配置到命令函数
            args.progress = progress
            args.config = config_overrides

            result = args.func(args)

            # 格式化输出
            output = OutputFormatter.format(result, mode=output_format)
            print(output)

            if known_args.verbose:
                print(json.dumps({
                    "command": args.command,
                    "status": "completed",
                    "output_format": output_format
                }), file=sys.stderr)

            sys.exit(0)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print(json.dumps(error_result("操作被用户中断", "KeyboardInterrupt")),
              file=sys.stderr)
        sys.exit(130)

    except FileNotFoundError as e:
        print(json.dumps(error_result(f"文件未找到: {e}", "FileNotFoundError")),
              file=sys.stderr)
        sys.exit(2)

    except ImportError as e:
        print(json.dumps(error_result(f"模块导入失败: {e}", "ImportError")),
              file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(json.dumps(error_result(str(e), type(e).__name__),
                         details={"command": args.command} if hasattr(args, 'command') else None),
              file=sys.stderr)
        sys.exit(1)


def __version__() -> str:
    """获取版本号"""
    try:
        from HydroArray import __version__
        return __version__
    except ImportError:
        return "0.1.0"


def completion_action() -> Callable:
    """生成 shell 自动补全代码"""
    def bash_completion():
        return '''
_hydro_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="--help --version --output --verbose --json --text --compact --config"
    subcmds="calc model metrics io update"

    case "${COMP_CWORD}" in
        1)
            COMPREPLY=($(compgen -W "${opts} ${subcmds}" -- "${cur}"))
            ;;
        2)
            case "${prev}" in
                calc)
                    COMPREPLY=($(compgen -W "runoff evaporation cross-section" -- "${cur}"))
                    ;;
                model)
                    COMPREPLY=($(compgen -W "list create run calibrate" -- "${cur}"))
                    ;;
                metrics)
                    COMPREPLY=($(compgen -W "calculate batch" -- "${cur}"))
                    ;;
                io)
                    COMPREPLY=($(compgen -W "read-table parse batch convert" -- "${cur}"))
                    ;;
            esac
            ;;
    esac
}
complete -F _hydro_completion hydro
'''
    return bash_completion


if __name__ == "__main__":
    main()