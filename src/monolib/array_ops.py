from typing import Any

import numpy as np

ArrayLike = Any


def linear_to_db(x: ArrayLike) -> ArrayLike:
    """Convert magnitude to dB"""
    return 20 * np.log10(x)


def db_to_linear(x: ArrayLike) -> ArrayLike:
    """Convert dB to magnitude"""
    return 10 ** (x / 20)


def power_to_db(x: ArrayLike) -> ArrayLike:
    """Convert power to dB"""
    return 10 * np.log10(x)


def db_to_power(x: ArrayLike) -> ArrayLike:
    """Convert dB to power"""
    return 10 ** (x / 10)


def normalize(x: ArrayLike) -> ArrayLike:
    return x / np.max(abs(x))
