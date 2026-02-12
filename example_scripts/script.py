# %%
import jax.numpy as jnp

from monolib._disp import display_collection, display_mono
from monolib.containers import (
    Datum,
    add_tags,
    collect,
    cmap,
    merge,
    datum,
    over_data,
    over_tags,
    partition,
)

x1 = datum(jnp.arange(3), {"do_amplification": True})
x2 = datum(jnp.arange(3), {"do_amplification": False})
x3 = datum(jnp.arange(3), {"do_amplification": True})

# %%

c = collect(x1, x2, x3, tags={"date": "2026-02-08 16:25", "title": "Untitled"})
display_collection(c)


# %%
def resample(m: Datum) -> Datum:
    return datum(m.data, 8000, m.tags)


def take_sin(m: Datum) -> Datum:
    return datum(m.xp.sin(m.data), 8000, m.tags)


pipeline = {
    "double": lambda m: over_data(m, lambda x: x * 2),
    "add_this": lambda m: add_tags(m, "add", "this"),
    "change_sample_rate": resample,
    "take_sin": take_sin,
}

for label, func in pipeline.items():
    do, dont = partition(c, lambda c: c.tags.get("do_amplification", False))
    do = cmap(func)(do)

    # record steps as meta_data
    do = cmap(over_tags)(do, lambda tags: {"steps": tags.get("steps", []) + [label]})

    c = merge(do, dont)

display_collection(c)
