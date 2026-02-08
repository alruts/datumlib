from monolib.containers import Mono
from typing import Any


def time(m: Mono) -> Any:
    """Return the time vector"""
    return m.xp.arange(len(m.data)) / m.sample_rate


def frequency(m: Mono) -> Any:
    """Return the frequency vector"""
    return m.xp.fft.rfftfreq(len(m.data), 1 / m.sample_rate)


def frequency_response(m: Mono) -> Any:
    """Return the Fourier transform of the signal"""
    return m.xp.fft.rfft(m.data)


def magnitude_response(m: Mono) -> Any:
    """Return the magnitude of the Fourier transform"""
    return m.xp.abs(frequency_response(m))


def phase(m: Mono) -> Any:
    """Return the phase of the Fourier transform"""
    return m.xp.angle(frequency_response(m))


def unwrapped_phase(m: Mono) -> Any:
    """Return the unwrapped phase of the Fourier transform"""
    return m.xp.unwrap(phase(m))


def power(m: Mono) -> Any:
    """Return the power of the Fourier transform"""
    return m.xp.abs(frequency_response(m)) ** 2


def duration(m: Mono) -> Any:
    """Return the length in seconds"""
    return len(m.data) / m.sample_rate
