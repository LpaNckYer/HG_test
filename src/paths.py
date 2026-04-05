"""
项目路径约定（相对仓库根目录）：

- output/   脚本生成的表格、图像等可交付结果
- logs/     文本运行日志（与终端 logging 配合）
- tmp/      中间运行产物、调试 CSV、可清空
- config/cases/  save_load 读写的 JSON 算例（上半 ``load_parameters``、下半 ``load_parameters_down``）
- data/     参考剖面等只读输入（若不存在则 ensure_dirs 会创建空目录便于放置文件）

脚本请在修改 cwd 之前调用 ``ensure_dirs()``，再用 ``output_path(...)`` / ``logs_path(...)``
得到绝对路径，避免散落的相对路径难以维护。

日志：首次 ``ensure_dirs()`` 会列出各目录绝对路径——若 root logging 尚未挂 handler（常见：
先 ``ensure_dirs`` 再配置文件日志），则写入 **stderr** 前缀 ``[paths]``；否则写入本模块
logger 的 **INFO**（名为 ``paths``，便于单独调级别）。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

__all__ = [
    "project_root",
    "output_dir",
    "logs_dir",
    "tmp_dir",
    "cases_dir",
    "data_dir",
    "ensure_dirs",
    "output_path",
    "logs_path",
    "tmp_path",
    "cases_path",
    "data_path",
    "relative_to_project",
]

_logger = logging.getLogger(__name__)

# 首次 ensure_dirs 时打印布局，避免每轮网格扫描刷屏
_layout_logged = False


def project_root() -> Path:
    """仓库根目录（包含 ``src/``、``scripts/`` 的目录）。"""
    return Path(__file__).resolve().parent.parent


def output_dir() -> Path:
    return project_root() / "output"


def logs_dir() -> Path:
    return project_root() / "logs"


def tmp_dir() -> Path:
    return project_root() / "tmp"


def cases_dir() -> Path:
    """算例目录：``config/cases``。"""
    return project_root() / "config" / "cases"


def data_dir() -> Path:
    return project_root() / "data"


def _join_under(base: Path, *parts: str | os.PathLike[str]) -> Path:
    p = base
    for x in parts:
        p = p / Path(x)
    return p


def output_path(*parts: str | os.PathLike[str]) -> Path:
    """``output/<parts...>`` 的绝对路径（不要求目录已存在）。"""
    return _join_under(output_dir(), *parts)


def logs_path(*parts: str | os.PathLike[str]) -> Path:
    """``logs/<parts...>`` 的绝对路径。"""
    return _join_under(logs_dir(), *parts)


def tmp_path(*parts: str | os.PathLike[str]) -> Path:
    """``tmp/<parts...>`` 的绝对路径。"""
    return _join_under(tmp_dir(), *parts)


def cases_path(*parts: str | os.PathLike[str]) -> Path:
    """``config/cases/<parts...>`` 的绝对路径（与 save_load 一致）。"""
    return _join_under(cases_dir(), *parts)


def data_path(*parts: str | os.PathLike[str]) -> Path:
    """``data/<parts...>`` 的绝对路径。"""
    return _join_under(data_dir(), *parts)


def relative_to_project(path: str | os.PathLike[str]) -> Path:
    """若 *path* 为相对路径，则按项目根解析；已是绝对路径则规范化后返回。"""
    p = Path(path)
    return p if p.is_absolute() else (project_root() / p).resolve()


def ensure_dirs(*, log_layout: bool = True) -> None:
    """
    创建标准目录（幂等）。

    *log_layout* 为 True 时，仅在进程内第一次调用时写一条 INFO（需已配置 logging）；
    便于确认日志与结果文件落在何处。
    """
    global _layout_logged
    root = project_root()
    mapping = {
        "output": output_dir(),
        "logs": logs_dir(),
        "tmp": tmp_dir(),
        "config/cases": cases_dir(),
        "data": data_dir(),
    }
    for d in mapping.values():
        d.mkdir(parents=True, exist_ok=True)

    if log_layout and not _layout_logged:
        _layout_logged = True
        lines = [f"  {name}: {path}" for name, path in mapping.items()]
        msg = "Standard project directories (under %s):\n%s" % (root, "\n".join(lines))
        root_log = logging.getLogger()
        if root_log.handlers:
            _logger.info(msg)
        else:
            # 多数脚本先 ensure_dirs 再配置 FileHandler；此处保证仍能看到目录布局
            print(f"[paths] Project root: {root}", file=sys.stderr)
            for name, path in mapping.items():
                print(f"[paths]   {name}: {path}", file=sys.stderr)
