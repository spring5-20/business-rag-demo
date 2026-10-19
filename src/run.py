"""本地开发启动脚本。

用法（在项目根目录执行）：

    uv run python src/run.py
    uv run python src/run.py --host 0.0.0.0 --port 8888
    uv run python src/run.py --no-reload
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_root_on_sys_path() -> None:
    """把项目根目录加入 ``sys.path``，保证可以导入 ``src.app.main``。"""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _configure_environment() -> None:
    """统一配置本地运行环境。"""
    # Windows 下强制使用 UTF-8，避免 .env 或日志中出现中文时被 GBK 解码报错。
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # 热重载时 Uvicorn 会启动子进程，通过 PYTHONPATH 把项目根目录传给子进程。
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="本地启动 FastAPI 开发服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8889, help="监听端口，默认 8889")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="关闭热重载（默认开启热重载）",
    )
    parser.add_argument(
        "--skip-infra",
        action="store_true",
        help="跳过 PostgreSQL / Redis 连接，方便在没有外部依赖时快速预览接口",
    )
    return parser.parse_args()


def main() -> None:
    """以本地开发模式启动 FastAPI 应用。"""
    args = _parse_args()
    _ensure_project_root_on_sys_path()
    _configure_environment()
    if args.skip_infra:
        os.environ["SKIP_INFRA_ON_START"] = "true"

    print(f"启动开发服务: http://{args.host}:{args.port}")
    print(f"交互式文档:   http://{args.host}:{args.port}/docs")
    print(f"健康检查:     http://{args.host}:{args.port}/api/v1/health")

    uvicorn.run(
        "src.app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    main()
