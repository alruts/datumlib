from pprint import pprint

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from monolib.containers import Mono, MonoCollection, collect, mono
from monolib.transforms import PipelineDict

s1 = mono(np.array([1, 2, 3]), sample_rate=8, tags={"operator": "Kasper"})
s2 = mono(np.array([4, 5, 6]), sample_rate=8, tags={"operator": "Jesper"})
s3 = mono(jnp.array([7, 8, 9]), sample_rate=8, tags={"operator": "Jonatan"})
s4 = mono(np.array([10, 11, 12]), sample_rate=8, tags={"operator": "Kasper"})

c = collect(s1, s2, s3, s4, tags={"experiment": "A"})
print(c)


def do_transformation(x: np.ndarray) -> np.ndarray:
    import time

    # simulate some work
    time.sleep(0.2)
    return x * 2


# Construct pipeline
pipe = PipelineDict[MonoCollection](
    {
        "add1": lambda s: s.apply(lambda x: x + 1),
        "mul2": lambda s: s.apply(lambda x: x * 2),
        "div3": lambda s: s.apply(lambda x: x / 3),
        "do_transformation": lambda s: s.apply(do_transformation),
    }
)

processed_c, scoped = pipe(c, scope=True)
breakpoint()


# Assign colors to steps
step_colors = {
    "add1": "red",
    "mul2": "orange",
    "div3": "b",
    "do_transformation": "k",
}

n_entries = len(processed_c.entries)
fig, axes = plt.subplots(n_entries, 1, figsize=(8, 3 * n_entries), squeeze=False)
for label, result in scoped.items():
    n_entries = len(result.valid_entries)

    for idx, m in enumerate(result.valid_entries):
        ax = axes[idx, 0]  # because squeeze=False gives 2D array of axes
        ax.plot(m.data, label=label, color=step_colors[label])
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")

plt.legend()
plt.tight_layout()
plt.show()


def classify_signal(m: Mono) -> dict:
    max_val = m.xp.max(m.data)
    if max_val > 15:
        label = "High"
    elif max_val > 10:
        label = "Medium"
    else:
        label = "Low"
    return {"class": label, "max_val": max_val}


labeled_c = processed_c.map_tags(classify_signal)
pprint(labeled_c)


high_c = labeled_c.filter(lambda x: x.tags.get("class") == "High")
pprint(high_c)

plt.figure(figsize=(10, 6))

# Assign colors to classes
class_colors = {
    "Low": "blue",
    "Medium": "orange",
    "High": "red",
}

for m in labeled_c.valid_entries:
    cls = m.tags["class"]
    plt.plot(m.data, marker="o", linestyle="-", color=class_colors[cls], label=cls)

# Create a legend without duplicates
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title("Mono Signal Visualization by Class")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
