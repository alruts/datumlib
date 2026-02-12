from logging import warning
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    TypeVar,
)

from typing_extensions import Concatenate, ParamSpec

P = ParamSpec("P")
R = TypeVar("R", bound="Mono")


class Mono(NamedTuple):
    """Data container for a 1D signal attached with relevant meta data. A `Mono`
    object is a NamedTuple which contains the following fields:

    - data (Any): The data to be stored in the container.
    - sample_rate (float): The sample rate of the data.
    - tags (dict): Meta data in the form of a dictionary. Default is an empty
    dictionary.

    ```
    >>> x = Mono(1.0, 1000, {"some_number": 42})
    >>> x
    Mono(data=1.0, sample_rate=1000, tags={'some_number': 42})
    >>> data, sample_rate, md = x
    >>> data, sample_rate, md
    (1.0, 1000, {'some_number': 42})
    >>> x.data # access via dot notation
    1.0
    >>> x.tags["some_number"]
    42
    >>> x.tags['x'] = 'x'
    ```

    """

    data: Any
    sample_rate: float | int
    tags: dict = {}

    @property
    def xp(self):
        """Return the array namespace for the underlying data."""
        if hasattr(self.data, "__array_namespace__"):
            return self.data.__array_namespace__()
        import numpy as np

        return np


class MonoCollection(NamedTuple):
    """Data container for multiple mono signals.

    A `MonoCollection` object is a NamedTuple which contains the following fields:
    - entries (`Tuple[Mono]`): A tuple of `Mono` objects.
    - tags (`dict`): Meta data in the form of a dictionary. Default is an empty
    dictionary.

    Mono collections can be constructed from multiple `Mono` objects using the
    `collect` function.

    >>> x = mono(1, 1)
    >>> y = mono(1, 1)
    >>> collect(x, y, tags={"date": "1999-12-31"})
    MonoCollection(
      entries=(
        Mono(data=1, sample_rate=1, tags={}),
        Mono(data=1, sample_rate=1, tags={}),
      ),
      tags={'date': '1999-12-31'}
    )

    """

    entries: tuple[Optional[Mono], ...]
    tags: dict = {}

    @property
    def valid_entries(self) -> list[Mono]:
        """Return a list of non-None Mono entries."""
        return [x for x in self.entries if x is not None]

    def __repr__(self) -> str:
        entry_lines = []
        for entry in self.entries:
            entry_lines.append(f"    {repr(entry)},")
        entries_repr = "\n".join(entry_lines)
        return (
            f"MonoCollection(\n"
            f"  entries=(\n{entries_repr}\n  ),\n"
            f"  tags={repr(self.tags)}\n"
            f")"
        )


# either
SignalContainer = Mono | MonoCollection


def mono(
    data: Any,
    sample_rate: float,
    tags: dict = {},
) -> Mono:
    """Create a mono signal container

    ```python
    >>> x = mono([1], 1, {"level": 42})
    >>> x
    Mono(data=[1], sample_rate=1, tags={'level': 42})
    >>> data, sample_rate, md = x # ucking
    >>> data, sample_rate, md
    ([1], 1, {'level': 42})
    >>> x.data # access via dot notation
    [1]

    ```

    ### Args:
    - data (`Any`): The data to be stored in the container.
    - sample_rate (`float`): The sample rate of the data.
    - tags (`dict`): Meta data in the form of a dictionary. Default is an empty
    dictionary.

    ### Returns:
    A `NamedTuple` containing the data, sample rate, and meta data.
    - data (`Any`): The data to be stored in the
    container.
    - sample_rate (`float`): The sample rate of the data.
    - tags (`dict`): The meta data dictionary.

    """
    return Mono(data, sample_rate, tags)


def collect(*monos: Optional[Mono], tags: dict = {}) -> MonoCollection:
    """Collect multiple `Mono` objects into a `MonoCollection` with optional
    meta data.

    ```python
    >>> x = mono([1], 1)
    >>> y = mono([1], 1)
    >>> collect(x, y, tags={"session": "0000-00-00"})
    MonoCollection(
      entries=(
        Mono(data=[1], sample_rate=1, tags={}),
        Mono(data=[1], sample_rate=1, tags={}),
      ),
      tags={'session': '0000-00-00'}
    )

    ```

    ### Args:
    - monos (`Mono`): One or more mono objects.
    - tags (`dict`): Meta data in the form of a dictionary. Keyword only argument.

    ### Returns:
    A `NamedTuple` containing a tuple of `Mono` objects and the meta data.
    - entries (`Tuple[Mono]`): A tuple of `Mono` objects.
    - tags (`dict`): The meta data dictionary.
    """

    return MonoCollection(monos, tags)


def partition(
    collection: MonoCollection,
    masking_func: Callable[[Mono], bool],
) -> tuple[MonoCollection, MonoCollection]:
    """Splits `collection` into two separate `MonoCollection` objects based on
    a predicate function.

    Each `Mono` in the original collection is tested with `masking_fn`. If the
    function returns `True`, the `Mono` is placed in the first output
    collection; otherwise, it is placed in the second. The position of each
    `Mono` is preserved, and unmatched entries are replaced with `None`.

    ```python
    >>> x = mono([1], 1, {"label": "a"})
    >>> y = mono([1], 1, {"label": "b"})
    >>> xy = collect(x, y)
    >>> filtered, rest = partition(xy, lambda x: x.tags.get("label") == "a")
    >>> filtered
    MonoCollection(
      entries=(
        Mono(data=[1], sample_rate=1, tags={'label': 'a'}),
        None,
      ),
      tags={}
    )
    >>> rest
    MonoCollection(
      entries=(
        None,
        Mono(data=[1], sample_rate=1, tags={'label': 'b'}),
      ),
      tags={}
    )

    ```
    ### Args:
    - collection (`MonoCollection`): The it collection of `Mono` objects.
    - masking_func (`Callable[[Mono], bool]`): A function that returns `True`
      for elements to include in the first output collection, and `False` for
      the second.

    ### Returns:

    A tuple of two `MonoCollection` object that preserve the structure of the
    it collection, with unmatched entries replaced by `None`.
    - The first contains elements where `masking_fn` returns `True`.
    - The second contains elements where it returns `False`."""

    filtered, rest = zip(
        *(
            (x, None) if x is None or masking_func(x) else (None, x)
            for x in collection.entries
        )
    )
    return MonoCollection(tuple(filtered), collection.tags), MonoCollection(
        tuple(rest), collection.tags
    )


# helper for merge
def _check_and_pick(xs: tuple[Mono, ...]):
    hits = [x for x in xs if x is not None]
    if len(hits) > 1:
        raise ValueError("`MonoCollection` has overlapping (mono) entries.")
    return hits[0] if hits else None


def merge(*collections: MonoCollection) -> MonoCollection:
    """Merges `collections` objects (typically from `partition`) into a single
    MonoCollection.

    This function assumes that each it collection shares the same structure,
    and that only one collection has a non-`None` entry at each position. This
    is typically used to reverse the effect of `partition`, after modifying the
    partitions separately.

    ```python
    >>> x = mono(1, 1, {"label": "a"})
    >>> y = mono(1, 1, {"label": "b"})
    >>> mc = collect(x, y)
    >>> filtered, rest = partition(mc, lambda x: x.tags.get("label") == "a")
    >>> merged = merge(
    ...     filtered.apply(lambda x: x + 1),
    ...     rest.apply(lambda x: x - 1)
    ... )  # apply different processing to each section
    >>> merged
    MonoCollection(
      entries=(
        Mono(data=2, sample_rate=1, tags={'label': 'a'}),
        Mono(data=0, sample_rate=1, tags={'label': 'b'}),
      ),
      tags={}
    )

    ```
    """
    entries = zip(*(c.entries for c in collections))
    merged = tuple(_check_and_pick(xs) for xs in entries)

    # check if all tags are the same
    hashable_dicts = {tuple(sorted(c.tags.items())) for c in collections}
    if len(hashable_dicts) != 1:
        warning(
            "Tags for collections to be merged do not match!, using tags from the first collection."
        )

    tags = collections[0].tags
    return MonoCollection(merged, tags=tags)


# maps


def over_data(func: Callable[[Any], Any]) -> Callable[[Mono], Mono]:
    """Apply `fn` to the data field of the `Mono` container and return a transformed
    container

    ```python
    >>> x = mono(1, 44100)
    >>> over_data(lambda x: x + 1)(x)
    Mono(data=2, sample_rate=44100, tags={})

    ```
    """

    def _over_data(m: Mono) -> Mono:
        new_data = func(m.data)
        return Mono(new_data, m.sample_rate, m.tags)

    return _over_data


def map_data(func: Callable[[Mono], Any]) -> Callable[[Mono], Mono]:
    """Apply `fn` to the whole `Mono` container and return a transformed
    container

    ```python
    >>> x = mono(1, 44100)
    >>> map_data(lambda x: x.data + 1)(x)
    Mono(data=2, sample_rate=44100, tags={})

    ```
    """

    def _map_data(m: Mono) -> Mono:
        new_data = func(m)
        return Mono(new_data, m.sample_rate, m.tags)

    return _map_data


def map_tags(func: Callable[[Mono], dict]) -> Callable[[Mono], Mono]:
    """Update the metadata field of `mono` using `fn`.

    ```python
    >>> x = mono([1, 2, 3], 44100, {})
    >>> x = map_tags(lambda m: {"n_samples": len(m.data)})(x)
    >>> x.tags
    {'n_samples': 3}

    ```
    """

    def _map_tags(m: Mono) -> Mono:
        new_meta = func(m)
        if not isinstance(new_meta, dict):
            raise TypeError(
                f"Function passed to `update` must return a dictionary, got {type(new_meta)}."  # noqa: E501
            )
        return Mono(m.data, m.sample_rate, m.tags | new_meta)

    return _map_tags


def over_tags(func: Callable[[dict], dict]) -> Callable[[Mono], Mono]:
    """Update the metadata field of `mono` using `fn`.

    ```python
    >>> x = mono([1, 2, 3], 44100, {"idx": 3})
    >>> x = over_tags(lambda d: {"idx": d["idx"] + 1})(x)
    >>> x
    Mono(data=[1, 2, 3], sample_rate=44100, tags={'idx': 4})

    ```
    """

    def _over_tags(m: Mono) -> Mono:
        new_meta = func(m.tags)
        if not isinstance(new_meta, dict):
            raise TypeError(
                f"Function passed to `update` must return a dictionary, got {type(new_meta)}."  # noqa: E501
            )
        return Mono(m.data, m.sample_rate, m.tags | new_meta)

    return _over_tags


def add_tags(m: Mono, key: str, value: Any) -> Mono:
    """Add tags to `mono`.

    ```python
    >>> x: Mono = mono([1, 2, 3], 44100, {})
    >>> x = add_tags(x, key="label", value="a")
    >>> x
    Mono(data=[1, 2, 3], sample_rate=44100, tags={'label': 'a'})

    ```
    """
    return Mono(m.data, m.sample_rate, m.tags | {key: value})


def cmap(
    func: Callable[Concatenate[Mono, P], R],
) -> Callable[Concatenate[MonoCollection, P], MonoCollection]:
    """Lift a mono transformation to operate over mono collections"""

    def _collection_map(
        collection: MonoCollection, *args: P.args, **kwargs: P.kwargs
    ) -> MonoCollection:
        return collect(
            *[
                func(x, *args, **kwargs) if x is not None else None
                for x in collection.entries
            ],
            tags=collection.tags,
        )

    return _collection_map


# collection only
def filter_collection(
    c: MonoCollection, predicate: Callable[[Mono], bool]
) -> MonoCollection:
    """Filter a mono collection by a predicate"""
    return collect(*[x for x in c.valid_entries if predicate(x)], tags=c.tags)


def get_tags(c: MonoCollection, key: str, fill_with=None) -> list:
    return [x.tags.get(key, fill_with) for x in c.valid_entries]


def group_by_tag(c: MonoCollection, key: str) -> dict[str, MonoCollection]:
    all_tags = get_tags(c, key)
    unique_tags = list(set(all_tags))
    return {
        tag: filter_collection(c, lambda m: m.tags.get(key) == tag)
        for tag in unique_tags
    }


def compact(c: MonoCollection) -> MonoCollection:
    return collect(*c.valid_entries, tags=c.tags)
