# mono utils
from typing import Any, Callable

from numpy.typing import ArrayLike

from monolib.containers import Mono


def _validate(x: Mono):
    if len(x.data.shape) > 1:
        raise ValueError(f"Data must be 1d time-series!, got shape: {x.data.shape}")


def time(x: Mono) -> ArrayLike:
    """Return the time vector"""
    _validate(x)
    return x.xp.arange(len(x.data)) / x.sample_rate


def frequency(x: Mono) -> ArrayLike:
    """Return the frequency vector"""
    _validate(x)
    return x.xp.fft.rfftfreq(len(x.data), 1 / x.sample_rate)


def frequency_response(x: Mono) -> ArrayLike:
    """Return the Fourier transform of the signal"""
    _validate(x)
    return x.xp.fft.rfft(x.data)


def magnitude_response(x: Mono) -> ArrayLike:
    """Return the magnitude of the Fourier transform"""
    _validate(x)
    return x.xp.abs(frequency_response(x))


def phase(x: Mono) -> ArrayLike:
    """Return the phase of the Fourier transform"""
    _validate(x)
    return x.xp.angle(frequency_response(x))


def unwrapped_phase(x: Mono) -> ArrayLike:
    """Return the unwrapped phase of the Fourier transform"""
    _validate(x)
    return x.xp.unwrap(phase(x))


def power(x: Mono) -> ArrayLike:
    """Return the power of the Fourier transform"""
    _validate(x)
    return x.xp.abs(frequency_response(x)) ** 2


def duration(x: Mono) -> ArrayLike:
    """Return the length in seconds"""
    _validate(x)
    return len(x.data) / x.sample_rate


def mono_lift(fn: Callable[[Any], ArrayLike]) -> Callable[[Mono], Mono]:
    """
    Lift a pure array function to operate on data field of Mono objects.
    """

    def mono_func(x: Mono, *args, **kwargs) -> Mono:
        data = getattr(x, "data")
        result = fn(data, *args, **kwargs)
        return result

    return mono_func
