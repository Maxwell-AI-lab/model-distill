"""日志工具"""

import logging
import sys
from pathlib import Path
from rich.logging import RichHandler


def setup_logger(name: str = "distill", level: str = "INFO", log_file: str = "") -> logging.Logger:
    """设置 Rich 日志"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Rich 控制台输出
    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)

    return logger
