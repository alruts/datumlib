"""
Room acoustics example.

We generate synthetic room impulse responses (RIRs),
extract common acoustical parameters, and visualize them.
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from monolib.acoustics import (
    early_to_late_index,
    early_to_total_index,
    edc,
    reverb_time,
)
from monolib.array_ops import normalize
from monolib.containers import Mono, collect, mono
from monolib.mono_utils import data_lift
from monolib.pipe import PipelineDict

# ---------------------------------------------------------------------
# Synthetic room impulse responses
# ---------------------------------------------------------------------

rng = np.random.default_rng(42)
fs = 48_000


sources: dict[str, tuple[float, float, float]] = {
    "S1: Corner": (0.0, 0.0, 0.0),
    "S2: Listener": (0.5, 0.5, 0.5),
}

receivers: dict[str, tuple[float, float, float]] = {
    "R1": (0.1, 0.2, 0.3),
    "R2": (0.6, 0.7, 0.5),
    "R3": (0.4, 0.2, 0.2),
    "R4": (0.1, 0.5, 0.7),
    "R5": (0.4, 0.5, 1.5),
}


def generate_rir(length=fs * 8, rt60=0.6, noise=1e-4):
    t = np.arange(length) / fs

    # amplitude decay consistent with RT60 definition
    decay = np.exp(-6.91 * t / rt60)

    rir = decay * rng.standard_normal(length)
    rir[0] = 1.0
    rir += noise * rng.standard_normal(length)

    return rir


# make decaying signals
signals = []
for (src_id, src_loc), (rec_id, rec_loc) in product(sources.items(), receivers.items()):
    rt60 = rng.uniform(0.3, 1.1)
    rir = generate_rir(rt60=rt60)

    # append metadata to each thing
    signals.append(
        mono(
            rir,
            sample_rate=fs,
            tags={
                "rt60_true": rt60,
                "source_id": src_id,
                "source_loc": src_loc,
                "receiver_id": rec_id,
                "receiver_loc": rec_loc,
            },
        )
    )

# collect into `MonoCollection`
collection = collect(*signals, tags={"dataset": "synthetic_rooms"})

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

# define a bunch of preprocessing functions on single signals


def truncate(m: Mono):
    # access via unpacking
    x, sample_rate, _ = m
    trunc_sec = 3
    return x[: sample_rate * trunc_sec]


# lift array function to operate on mono inputs
normalize_func = data_lift(normalize)

# define a preprocessing pipeline. This will sequentially process
# the signals in `collection` and return an updated collection
pipeline = PipelineDict(
    {
        "normalize": normalize_func,
        "truncate": truncate,
    }
)

# Process collection via pipeline and visualize progress
collection = pipeline(collection, progress_meter=True)


# ---------------------------------------------------------------------
# Room acoustics feature extraction
# ---------------------------------------------------------------------


# define a function to extract various stats about our signals
def extract_acoustics(m: Mono):
    e = m.transform_data(lambda x: edc(x.data))
    rt60 = reverb_time(collect(e))
    d50 = early_to_total_index(m, 50e-3)
    c80 = early_to_late_index(m, 80e-3)
    return {
        "rt60": rt60,
        "d50": d50,
        "c80": c80,
    }


# Add acoustics parameters as tags to each signal
collection = collection.map_tags(lambda m: extract_acoustics(m))


# ---------------------------------------------------------------------
# Collect features
# ---------------------------------------------------------------------

# this kind of thing def needs to be more convenient
RT60 = collection.get_tags("rt60")
RT60_true = collection.get_tags("rt60_true")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

styles = iter(["o", "x"])

for source_id, collection in collection.group_by("source_id").items():
    rec_locs = collection.get_tags("receiver_loc")
    c80 = collection.get_tags("c80")

    rx, ry, rz = zip(*rec_locs)
    sc = ax.scatter(rx, ry, rz, c=c80, cmap="viridis", s=80, marker=next(styles))

# mark sources too
for src_id, src_loc in sources.items():
    ax.scatter(*src_loc, s=150, marker="*", label=src_id)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("C80 by Receiver Location")

plt.colorbar(sc, label="C80 (dB)")
plt.legend()
plt.tight_layout()
plt.show()

# rt vs truth
plt.figure(figsize=(7, 6))
plt.scatter(RT60_true, RT60)
plt.plot([0, 1.5], [0, 1.5], "--", lw=1)
plt.xlabel("True RT60 [s]")
plt.ylabel("Estimated RT60 [s]")
plt.title("RT60 Estimation from Synthetic RIRs")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
