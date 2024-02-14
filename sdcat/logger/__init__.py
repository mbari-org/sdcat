# sdcat, Apache-2.0 license
# Filename: sdcat/logger/__init__.py
# Description:  Logs to both a file and the console. The file is named with the current date.

import logging
from pathlib import Path
from datetime import datetime as dt

LOGGER_NAME = "sdcat"
DEBUG = True


class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(_Singleton('SingletonMeta', (object,), {})): pass


class CustomLogger(Singleton):
    logger = None
    summary_df = None
    output_path = Path.cwd()

    def __init__(self, output_path: Path = Path.cwd(), output_prefix: str = "uav"):
        """
        Initialize the logger
        """
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        self.output_path = output_path
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        # default log file date to today
        now = dt.utcnow()

        # log to file
        self.log_filename = output_path / f"{output_prefix}_{now:%Y%m%d}.log"
        handler = logging.FileHandler(self.log_filename, mode="w")
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        self.logger.info(f"Logging to {self.log_filename}")

    def loggers(self) -> logging.Logger:
        return self.logger

def create_logger_file(log_path: Path, prefix: str=""):
    """
    Create a logger file
    :param log_path: Path to the log file
    """
    # create the log directory if it doesn't exist
    log_path.mkdir(parents=True, exist_ok=True)
    return CustomLogger(log_path, f'{LOGGER_NAME}{prefix}')


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
