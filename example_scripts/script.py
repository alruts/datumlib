# %%
from monolib.containers import (
    Mono,
    mono,
    collect,
    collection_map,
    partition,
    merge,
    over_data,
    over_tags,
    add_tags,
)
from monolib._disp import display_collection, display_mono
import jax.numpy as jnp

x1 = mono(jnp.arange(3), 48000, {"do_amplification": True})
x2 = mono(jnp.arange(3), 48000, {"do_amplification": False})
x3 = mono(jnp.arange(3), 48000, {"do_amplification": True})

display_mono(x1)
# %%

c = collect(x1, x2, x3)
display_collection(c)


# %%
def resample(m: Mono) -> Mono:
    return mono(m.data, 8000, m.tags)


def take_sin(m: Mono) -> Mono:
    return mono(m.xp.sin(m.data), 8000, m.tags)


pipeline = {
    "double": lambda m: over_data(m, lambda x: x * 2),
    "add_this": lambda m: add_tags(m, "add", "this"),
    "change_sample_rate": resample,
    "take_sin": take_sin,
}

for label, func in pipeline.items():
    do, dont = partition(c, lambda c: c.tags.get("do_amplification", False))
    do = collection_map(func)(do)

    # record steps as meta_data
    do = collection_map(over_tags)(
        do, lambda tags: {"steps": tags.get("steps", []) + [label]}
    )

    c = merge(do, dont)

display_collection(c)


# -- the next step of this library is to make some clever pipelines
# -- perhaps I can use this 'save steps' trick as optional
# -- perhaps I can have tqdm stuff
# -- perhaps I can have the 'stepped' dictionary as a thing


# -- other helper for collections are 'groupby' and 'filter' which would allow
# arbitrary partitions of a collection, which might be useful
# for plotting

# -- other might be 'compact' which would get rid of none entries
