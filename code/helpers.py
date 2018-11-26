# -*- coding: utf -*-

"""This module contains general "helper" methods and variables."""

import functools
import logging
import os

CODEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.dirname(CODEDIR)
IMGDIR = os.path.join(BASEDIR, "img")


def log_func_edges(func):
    """Decorator to log entering/exiting of functions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Generic wrapper function."""
        logging.debug(f"Initializing processing for `{func.__name__}`...")
        results = func(*args, **kwargs)
        logging.debug(f"Finalizing processing for `{func.__name__}`...")
        return results

    return wrapper


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s:L%(lineno)s - %(message)s",
    )
