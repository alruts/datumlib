from monolib.containers import Datum
from typing import Any


def _get_sample_rate(d: Datum):
    if this := d.tags.get("sample_rate"):
        return this
    else:
        raise KeyError(
            "Attempted to access tags['sample_rate'], "
            "but no sample rate is set for this Datum."
        )


def time(d: Datum) -> Any:
    """Return the time vector"""
    sample_rate = _get_sample_rate(d)
    return d.xp.arange(len(d.data)) / sample_rate


def frequency(d: Datum) -> Any:
    """Return the frequency vector"""
    sample_rate = _get_sample_rate(d)
    return d.xp.fft.rfftfreq(len(d.data), 1 / sample_rate)


def frequency_response(d: Datum) -> Any:
    """Return the Fourier transform of the signal"""
    return d.xp.fft.rfft(d.data)


def magnitude_response(d: Datum) -> Any:
    """Return the magnitude of the Fourier transform"""
    return d.xp.abs(frequency_response(d))


def phase(d: Datum) -> Any:
    """Return the phase of the Fourier transform"""
    return d.xp.angle(frequency_response(d))


def unwrapped_phase(d: Datum) -> Any:
    """Return the unwrapped phase of the Fourier transform"""
    return d.xp.unwrap(phase(d))


def power(d: Datum) -> Any:
    """Return the power of the Fourier transform"""
    return d.xp.abs(frequency_response(d)) ** 2


def duration(d: Datum) -> Any:
    """Return the length in seconds"""
    sample_rate = _get_sample_rate(d)
    return len(d.data) / sample_rate
