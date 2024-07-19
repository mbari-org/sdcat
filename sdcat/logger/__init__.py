# sdcat, Apache-2.0 license
# Filename: sdcat/logger/__init__.py
# Description:  Logs to both a file and the console. The file is named with the current date.

import logging
import os
from pathlib import Path
from datetime import datetime as dt, timezone

LOGGER_NAME = "sdcat"
DEBUG = True


class _Singleton(type):
    """A metaclass that creates a Singleton base class when called."""

    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(_Singleton("SingletonMeta", (object,), {})):
    pass


class CustomLogger(Singleton):
    logger = None
    output_path = Path.home() / "sdcat" / "logs"

    def __init__(self, output_path: Path = Path.cwd(), output_prefix: str = LOGGER_NAME):
        """
        Initialize the logger
        """
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.setLevel(logging.DEBUG)
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        # default log file date to today
        now = dt.now(timezone.utc)

        # log to file
        self.log_filename = output_path / f"{output_prefix}_{now:%Y%m%d}.log"
        handler = logging.FileHandler(self.log_filename, mode="w")
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        # also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        self.logger.info(f"Logging to {self.log_filename}")

    def loggers(self) -> logging.Logger:
        return self.logger


def create_logger_file(prefix: str = "sdcat"):
    """
    Create a logger file
    :param log_path: Path to the log file
    """
    ENVIRONMENT = str(os.getenv("ENVIRONMENT"))
    if ENVIRONMENT and ENVIRONMENT.upper() == "TESTING":
        log_path = Path("logs")
    else:
        log_path = Path.home() / "sdcat" / "logs"
        # Check if can write to the log path, and if not revert to system temp
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            test_file = log_path / "test.txt"
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()
        except PermissionError:
            import tempfile

            temp_dir = tempfile.gettempdir()
            log_path = Path(temp_dir) / "sdcat" / "logs"

    # create the log directory if it doesn't exist
    log_path.mkdir(parents=True, exist_ok=True)
    return CustomLogger(log_path, prefix)


def custom_logger() -> logging.Logger:
    """
    Get the logger
    """
    return logging.getLogger(LOGGER_NAME)


def err(s: str):
    custom_logger().error(s)


def info(s: str):
    custom_logger().info(s)


def debug(s: str):
    custom_logger().debug(s)


def warn(s: str):
    custom_logger().warning(s)


def exception(s: str):
    custom_logger().exception(s)


def critical(s: str):
    custom_logger().critical(s)
