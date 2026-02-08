import logging
import os
import sys
from datetime import datetime


def setup_logging(log_dir='logs'):
    """
    Sets up the root logger to log to a file and to the console.
    The log file will be named with a timestamp in the specified directory.
    This function should be called once at the beginning of the application.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    log_file_path = os.path.join(log_dir, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file_path

def get_logger(name):
    """
    Returns a logger instance with the specified name.
    The logger's name will be used as a prefix in the log messages
    as configured by setup_logging.
    """
    return logging.getLogger(name)