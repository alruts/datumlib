from typing import Callable, Optional

import numpy as np
import scipy.signal as signal
from numpy.polynomial.polynomial import Polynomial

from monolib.array_ops import normalize, power_to_db
from monolib.comp import pipe
from monolib.containers import (
    Mono,
    MonoCollection,
    collect,
)
from monolib.mono_utils import (
    time,
)


def _find_x_for_y(p_std: Polynomial, y_target: float) -> float:
    """p_std is in the form Polynomial([a0, a1]) => y = a0 + a1*x"""
    a0, a1 = p_std.coef
    if a1 == 0:
        raise ValueError("Slope is zero; cannot solve for x.")
    x_target = (y_target - float(a0)) / float(a1)
    return x_target


def reverb_time(edcs: MonoCollection, decay_range=30, decay_target=60):
    """Assume MonoCollection of edc curves"""
    # Fit a line between -5 and -(decay_target + 5), and evaluate at -decay_target
    interval = (-5, -decay_range - 5)
    fits = [
        fit_rt_line(x=time(curve), y=curve.data, interval=interval)
        for curve in edcs.valid_entries
    ]
    # extrapolate to decay_target
    rts = [_find_x_for_y(fit, -decay_target) for fit in fits]
    return list(map(float, rts))


def fit_rt_line(
    x: np.ndarray,
    y: np.ndarray,
    interval: Optional[tuple[float, float]],
) -> Polynomial:
    """..."""

    # filter data outside of domain
    if interval is not None:
        fst, snd = interval
        mask = (y <= fst) & (y >= snd)
        x = x[mask]
        y = y[mask]

    # linear least-squares fit
    p = Polynomial.fit(x, y, 1)

    # Convert to standard polynomial basis
    p_std = p.convert()

    return p_std


def make_octave_filter_bank(
    center_frequencies: list[float], octave_fraction: int
) -> Callable[[Mono], MonoCollection]:
    """
    Creates a filter bank composed of bandpass filters spaced by a given octave
    fraction.

    This function returns a callable that takes a mono signal and returns a collection
    of filtered signals, each corresponding to one of the specified center frequencies.
    The filters are designed based on octave band spacing, where each band covers
    a frequency range proportional to an octave fraction.

    """

    def _bank(x: Mono):
        filter_bank = []

        for cf in center_frequencies:
            band_ratio = 2 ** (1 / (2 * octave_fraction))
            low = cf / band_ratio
            high = cf * band_ratio

            sos = signal.butter(
                4, [low, high], btype="band", fs=x.sample_rate, output="sos"
            )

            # Use sos as default value to avoid late binding issue
            filter_bank.append(lambda x, sos=sos: signal.sosfiltfilt(sos, x.data))

        xs = [x.map_data(f) for f in filter_bank]

        return collect(*xs)

    return _bank


def edc(x: np.ndarray):
    """
    Computes the Energy Decay Curve (EDC) for a mono signal.

    Parameters:
    - x: Mono signal

    Returns:
    - edc_db: Energy decay curve in dB (normalized to 0 dB)
    """
    # Calculate energy
    energy = x**2

    # Schroeder integration
    edc = np.cumsum(energy[::-1])[::-1]

    # Normalize and convert to dB
    edc_db = pipe(normalize, power_to_db)(edc)

    return edc_db


def early_to_late_index(x: Mono, transition_time: float):
    """c80 for 80e-3"""
    t = time(x)
    transition_idx = np.searchsorted(t, transition_time)

    # Calculate energy
    energy = x.data**2

    # Split at transition time
    head = energy[:transition_idx]
    tail = energy[transition_idx:]

    # Forward-integration
    ratio = np.sum(head) / np.sum(tail)
    ratio_db = power_to_db(ratio)

    return float(ratio_db)


def early_to_total_index(x: Mono, transition_time: float):
    """d50 for transition_time 50e-3"""
    t = time(x)
    transition_idx = np.searchsorted(t, transition_time)

    # Calculate energy
    energy = x.data**2

    # Split at transition time
    head = energy[:transition_idx]

    # Forward-integration
    ratio = np.sum(head) / np.sum(energy)
    ratio_db = 10 * np.log10(ratio)

    return float(ratio_db)
