"""基于 loguru 的日志配置：控制台 + 文件（按大小轮转）。"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from qkquant.config import get_settings

_configured = False


def setup_logger(level: str | None = None) -> None:
    """初始化 loguru。幂等：重复调用只生效一次。"""
    global _configured
    if _configured:
        return

    cfg = get_settings().logging
    effective_level = (level or cfg.level).upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=effective_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        enqueue=False,
    )

    log_file: Path = cfg.file_abs
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_file),
        level=effective_level,
        rotation=cfg.rotation,
        retention=cfg.retention,
        encoding="utf-8",
        enqueue=True,
    )

    _configured = True


__all__ = ["logger", "setup_logger"]
