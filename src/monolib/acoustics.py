from typing import Callable, Optional

import numpy as np
import scipy.signal as signal
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import butter, lfilter

from monolib.array_ops import normalize, power_to_db
from monolib.comp import pipe
from monolib.containers import (
    Mono,
    MonoCollection,
    collect,
    mono,
)
from monolib.mono_utils import (
    frequency,
    frequency_response,
    mono_lift,
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

        xs = [x.transform_data(f) for f in filter_bank]

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


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def band_limited_noise(length, fs, lowcut, highcut):
    noise = np.random.randn(length)
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, noise)


def generate_rir(fs=16000, duration=1.0, rt60=0.5):
    """
    Generate a more realistic synthetic room impulse response.

    Parameters:
    - fs (int): Sampling rate
    - duration (float): Duration in seconds
    - rt60 (float): Approximate RT60 decay time (in seconds)

    Returns:
    - rir (numpy.ndarray): Generated RIR
    """
    n_samples = int(fs * duration)
    rir = np.zeros(n_samples)

    # Direct sound at sample 0
    rir[0] = 1.0

    # Early reflections (random delays and amplitudes)
    n_early = 10
    delays = np.random.randint(int(0.001 * fs), int(0.03 * fs), size=n_early)  # 1–30 ms
    amplitudes = np.random.uniform(0.3, 0.9, size=n_early) * np.exp(
        -delays / (fs * 0.05)
    )  # decay over time

    for d, a in zip(delays, amplitudes):
        if d < n_samples:
            rir[d] += a * np.random.choice([-1, 1])

    # Late reverberation: exponentially decaying band-limited noise
    tail_start = int(0.03 * fs)
    tail = np.random.randn(n_samples - tail_start)
    decay = np.exp(-np.linspace(0, duration - tail_start / fs, len(tail)) / rt60)

    # Optional: make tail frequency dependent (high frequencies decay faster)
    tail = band_limited_noise(len(tail), fs, 200, 6000) * decay

    rir[tail_start:] += 0.2 * tail

    # Normalize
    rir /= np.max(np.abs(rir) + 1e-9)

    return rir


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    center_frequencies = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]

    ir = mono(generate_rir(16000, duration=1, rt60=0.1), 16000)
    filter_bank = make_octave_filter_bank(center_frequencies, 1)

    # apply fb and tag with center freqs
    collection = filter_bank(ir).with_tags(center_frequencies, "center_frequency")

    # loop thru entries and plot
    for ir in collection.valid_entries:
        plt.plot(
            frequency(ir),
            frequency_response(ir),
            alpha=0.5,
            label=ir.tags.get("center_frequency"),
        )
    plt.ylim(-20, 20)
    plt.xlim(0, 3000)
    plt.legend()
    plt.show()

    # now apply edc function on the data
    edc_collection = collection.map_data(mono_lift(edc))

    for edc in edc_collection.valid_entries:
        plt.plot(time(edc), edc.data, alpha=0.5, label=edc.tags.get("center_frequency"))

    plt.legend()
    plt.ylim(-100, 0)
    plt.show()

    def compute_metrics(x):
        """return dict of line and t30"""
        t20_line = fit_rt_line(time(x), x.data, interval=(-5, -25))
        t20 = _find_x_for_y(t20_line, -60)

        t30_line = fit_rt_line(time(x), x.data, interval=(-5, -35))
        t30 = _find_x_for_y(t30_line, -60)

        c50 = early_to_late_index(x, 50e-3)

        return {
            "t20": t20,
            "t30": t30,
            "t30_line": t30_line,
            "t20_line": t20_line,
            "c50": c50,
        }

    # add metrics to the tags
    edc_collection = edc_collection.map_tags(compute_metrics)

    for ir in edc_collection.valid_entries:
        # plot the edc
        t = time(ir)
        plt.plot(t, ir.data)

        # plot fits for t20 and t30
        t20_line = ir.tags.get("t20_line", np.nan)
        t30_line = ir.tags.get("t30_line", np.nan)
        plt.plot(t, t20_line(t), "b--")
        plt.plot(t, t30_line(t), "r--")

        plt.ylim(-60, 0)
        rts = [
            round(ir.tags.get("t20", np.nan), 2),
            round(ir.tags.get("t30", np.nan), 2),
        ]
        plt.xticks(rts)
        plt.title(f"CF: {ir.tags.get('center_frequency', 'missing')} (Hz)")
        plt.show()
