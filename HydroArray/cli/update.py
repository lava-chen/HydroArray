"""更新相关命令"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any, Dict

from .output import success_result, error_result


def register(subparsers: argparse.ArgumentParser) -> None:
    """注册 update 子命令

    版本更新模块，包括：
    - check: 检查更新
    - install: 安装最新版本
    - info: 显示版本信息
    """
    update_parser = subparsers.add_parser(
        "update",
        help="版本更新与信息",
        description="检查和安装 HydroArray 更新",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查当前版本和可用更新
  hydro update check

  # 安装最新稳定版
  hydro update install

  # 安装特定版本
  hydro update install --version 0.2.0

  # 从 GitHub main 分支安装
  hydro update install --branch main

  # 显示版本信息
  hydro update info
        """
    )

    sub = update_parser.add_subparsers(dest="update_action", required=True,
                                      help="更新操作")

    # 检查更新
    _register_check(sub)

    # 安装更新
    _register_install(sub)

    # 版本信息
    _register_info(sub)


def _register_check(sub) -> None:
    """注册检查更新子命令"""
    p = sub.add_parser(
        "check",
        help="检查更新",
        description="检查是否有可用更新"
    )
    p.add_argument(
        "--pre-release",
        action="store_true",
        help="也检查预发布版本 (beta, alpha)"
    )
    p.add_argument(
        "--pip",
        action="store_true",
        help="使用 pip 而非 uv (如果有)"
    )
    p.set_defaults(func=_check_update)


def _register_install(sub) -> None:
    """注册安装更新子命令"""
    p = sub.add_parser(
        "install",
        help="安装更新",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="安装 HydroArray 更新",
        epilog="""
安装模式:
  - latest: 安装 PyPI 最新版本 (默认)
  - version: 安装指定版本
  - branch: 从 GitHub 分支安装
  - local: 从本地目录安装
        """
    )
    p.add_argument(
        "--mode", "-m",
        choices=["latest", "version", "branch", "local"],
        default="latest",
        help="安装模式 (默认: latest)"
    )
    p.add_argument(
        "--version", "-v",
        help="指定版本号 (如: 0.2.0)"
    )
    p.add_argument(
        "--branch", "-b",
        default="main",
        help="GitHub 分支名 (默认: main)"
    )
    p.add_argument(
        "--path", "-p",
        help="本地目录路径"
    )
    p.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新安装 (--force-reinstall)"
    )
    p.add_argument(
        "--pip",
        action="store_true",
        help="使用 pip 而非 uv"
    )
    p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过确认提示"
    )
    p.set_defaults(func=_install_update)


def _register_info(sub) -> None:
    """注册版本信息子命令"""
    p = sub.add_parser(
        "info",
        help="版本信息",
        description="显示详细的版本信息"
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="显示所有信息，包括依赖版本"
    )
    p.set_defaults(func=_show_info)


def _get_current_version() -> str:
    """获取当前版本号"""
    try:
        from HydroArray import __version__
        return __version__
    except ImportError:
        return "unknown"


def _get_package_info() -> Dict[str, Any]:
    """获取包信息"""
    info = {
        "current_version": _get_current_version(),
        "package_name": "HydroArray",
    }

    # 尝试从 pip/uv 获取信息
    try:
        result = subprocess.run(
            ["pip", "show", "HydroArray"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace("_", "-")
                    if key == "location":
                        info["install-location"] = value.strip()
                    elif key == "required-python":
                        info["python-version"] = value.strip()
    except Exception:
        pass

    return info


def _check_pypi_version(prerelease: bool = False) -> Dict[str, Any]:
    """检查 PyPI 上的最新版本"""
    import json

    try:
        result = subprocess.run(
            ["pip", "index", "versions", "HydroArray"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # 解析 pip index versions 输出
            for line in result.stdout.split("\n"):
                if "Available versions:" in line:
                    versions = line.split("Available versions:")[1].strip()
                    version_list = [v.strip() for v in versions.split(",")]
                    latest = version_list[-1] if version_list else None
                    return {"latest": latest, "available": version_list}
    except Exception:
        pass

    # 备选：通过 PyPI API
    try:
        import urllib.request
        url = "https://pypi.org/pypi/HydroArray/json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            latest = data["info"]["version"]
            return {"latest": latest, "url": data["info"]["project_url"]}
    except Exception:
        return {"error": "无法连接到 PyPI"}


def _check_update(args) -> Dict[str, Any]:
    """检查更新"""
    current = _get_current_version()
    pypi = _check_pypi_version(args.pre_release)

    result = {
        "current_version": current,
        "pypi_version": pypi.get("latest"),
        "up_to_date": False,
        "update_available": False,
    }

    if "error" not in pypi and current != "unknown":
        if pypi["latest"]:
            from packaging import version
            try:
                result["up_to_date"] = version.parse(current) >= version.parse(pypi["latest"])
                result["update_available"] = not result["up_to_date"]
            except Exception:
                result["update_available"] = current != pypi["latest"]

    if "error" in pypi:
        result["error"] = pypi["error"]

    return success_result(
        data=result,
        message="当前已是最新版本" if result["up_to_date"] else "发现新版本可用"
    )


def _install_update(args) -> Dict[str, Any]:
    """安装更新"""
    from pathlib import Path

    cmd = []

    # 确定使用 pip 还是 uv
    if args.pip:
        cmd = ["pip"]
    else:
        # 优先使用 uv
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            cmd = ["uv", "pip", "install"]
        except Exception:
            cmd = ["pip", "install"]

    # 添加 --force-reinstall 如果需要
    if args.force:
        cmd.append("--force-reinstall")

    # 根据模式确定包名
    if args.mode == "latest":
        cmd.extend(["-U", "HydroArray"])
    elif args.mode == "version":
        if not args.version:
            return error_result("缺少 --version 参数", "UsageError")
        cmd.extend(["-U", f"HydroArray=={args.version}"])
    elif args.mode == "branch":
        cmd.extend(["git+https://github.com/lava-chen/HydroArray.git",
                   f"@{args.branch}#egg=HydroArray"])
    elif args.mode == "local":
        if not args.path:
            return error_result("缺少 --path 参数", "UsageError")
        local_path = Path(args.path)
        if not local_path.exists():
            return error_result(f"本地路径不存在: {args.path}", "FileNotFoundError")
        cmd.extend(["-e", str(local_path.absolute())])

    # 确认提示
    if not args.yes and sys.stdout.isatty():
        confirm = input(f"执行: {' '.join(cmd)}\n确认安装? [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            return error_result("安装已取消", "Cancelled")

    # 执行安装
    try:
        print(f"执行命令: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 分钟超时
        )

        if result.returncode == 0:
            new_version = _get_current_version()
            return success_result(
                data={"version": new_version, "command": " ".join(cmd)},
                message=f"安装成功，当前版本: {new_version}"
            )
        else:
            return error_result(
                f"安装失败: {result.stderr}",
                "InstallError"
            )

    except subprocess.TimeoutExpired:
        return error_result("安装超时 (超过 5 分钟)", "Timeout")
    except Exception as e:
        return error_result(f"安装过程出错: {e}", "InstallError")


def _show_info(args) -> Dict[str, Any]:
    """显示版本信息"""
    info = _get_package_info()

    if args.all:
        info["dependencies"] = {}

        # 获取主要依赖版本
        try:
            result = subprocess.run(
                ["pip", "show", "numpy", "pandas", "xarray"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split("\n"):
                if ":" in line and "Version:" in line:
                    pkg, ver = line.split(":", 1)
                    info["dependencies"][pkg.strip().lower()] = ver.strip()
        except Exception:
            pass

    # 添加 Git 信息（如果有）
    try:
        git_dir = Path(__file__).parent.parent.parent / ".git"
        if git_dir.exists():
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=git_dir.parent
            )
            if result.returncode == 0:
                info["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    return success_result(
        data=info,
        message="HydroArray 版本信息"
    )