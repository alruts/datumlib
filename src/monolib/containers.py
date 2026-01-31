from typing import Any, Callable, NamedTuple, Optional, Sequence


class Mono(NamedTuple):
    """Data container for a mono signal. A `Mono` object is a NamedTuple which
    contains the following fields:

    - data (Any): The data to be stored in the container.
    - sample_rate (float): The sample rate of the data.
    - tags (dict): Meta data in the form of a dictionary. Default is an empty
    dictionary.

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

    """

    data: Any
    sample_rate: float | int
    tags: dict = {}

    def apply(self, fn):
        """Apply `fn` to the data field of the `Mono` container and return a transformed
        container

        ```python
        >>> x = mono(1, 44100)
        >>> x.apply(lambda x: x + 1)
        Mono(data=2, sample_rate=44100, tags={})

        ```
        """
        return Mono(fn(self.data), self.sample_rate, self.tags)

    def map_tags(self, fn: Callable[["Mono"], dict]) -> "Mono":
        """
        Update the metadata field of `mono` using `fn`.

        ```python
        >>> x: Mono = mono([1, 2, 3], 44100, {})
        >>> x = x.map_tags(lambda m: {"n_samples": len(m.data)})
        >>> x.tags
        {'n_samples': 3}

        ```
        """
        new_meta = fn(self)
        if not isinstance(new_meta, dict):
            raise TypeError(
                f"Function passed to `update` must return a dictionary, got {type(new_meta)}."  # noqa: E501
            )
        return Mono(self.data, self.sample_rate, self.tags | new_meta)

    def with_tags(self, value: Any, key: str) -> "Mono":
        """
        Update the metadata field of each Mono entry using fn.

        ```python
        >>> x: Mono = mono([1, 2, 3], 44100, {})
        >>> x = x.with_tags("a", key="label")
        >>> x
        Mono(data=[1, 2, 3], sample_rate=44100, tags={'label': 'a'})

        ```
        """
        return Mono(self.data, self.sample_rate, self.tags | {key: value})

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

    def apply(self, fn, progress_meter=False):
        """Apply `fn` to the data field of each `Mono` container and return a
        transformed collection, with a progress bar.

        ### Example:

        >>> x = mono(1, 1)
        >>> xx = collect(x, x)
        >>> xx.apply(lambda x: x + 1)
        MonoCollection(
          entries=(
            Mono(data=2, sample_rate=1, tags={}),
            Mono(data=2, sample_rate=1, tags={}),
          ),
          tags={}
        )

        """
        if progress_meter:
            from tqdm import tqdm

            transformed_entries = [
                x.apply(fn) if x else None
                for x in tqdm(self.entries, desc="Processing", unit="entry")
            ]
        else:
            transformed_entries = [x.apply(fn) if x else None for x in self.entries]
        return collect(*transformed_entries, tags=self.tags)

    @property
    def valid_entries(self) -> list[Mono]:
        """Return a list of non-None Mono entries."""
        return [x for x in self.entries if x is not None]

    def map_tags(self, fn: Callable[[Mono], dict]) -> "MonoCollection":
        """
        Update the metadata field of each `Mono` entry using `fn`.

        ```python
        >>> x: Mono = mono([1, 2, 3], 44100, {})
        >>> xs: MonoCollection = collect(x, x)
        >>> xs = xs.map_tags(lambda c: {"n_samples": len(c.data)})
        >>> xs
        MonoCollection(
          entries=(
            Mono(data=[1, 2, 3], sample_rate=44100, tags={'n_samples': 3}),
            Mono(data=[1, 2, 3], sample_rate=44100, tags={'n_samples': 3}),
          ),
          tags={}
        )

        ```
        """
        return collect(*[x_.map_tags(fn) for x_ in self.valid_entries], tags=self.tags)

    def with_tags(
        self: "MonoCollection", sequence: Sequence, key: str
    ) -> "MonoCollection":
        """
        Update the metadata field of each Mono entry using fn.

        ```python
        >>> x: Mono = mono([1, 2, 3], 44100, {})
        >>> x: MonoCollection = collect(x, x)
        >>> x = x.with_tags(["a", "b"], key="label")
        >>> x
        MonoCollection(
          entries=(
            Mono(data=[1, 2, 3], sample_rate=44100, tags={'label': 'a'}),
            Mono(data=[1, 2, 3], sample_rate=44100, tags={'label': 'b'}),
          ),
          tags={}
        )

        ```
        """
        return collect(
            *[
                x_.with_tags(val, key=key)
                for x_, val in zip(self.valid_entries, sequence)
            ],
            tags=self.tags,
        )

    def with_collection_tags(self, value: Any, key: str) -> "MonoCollection":
        """
        Update the metadata of the MonoCollection itself (not individual entries).

        ```python
        >>> x = collect(mono([1, 2], 44100), mono([3, 4], 44100))
        >>> x = x.with_collection_tags("experiment_1", key="experiment")
        >>> x.tags
        {'experiment': 'experiment_1'}

        ```
        """
        return MonoCollection(self.entries, self.tags | {key: value})

    def map_collection_tags(
        self, fn: Callable[[dict, "MonoCollection"], dict]
    ) -> "MonoCollection":
        """
        Update the metadata of the MonoCollection itself (not individual entries) using a function.

        The function `fn` receives the current tags dict **and the MonoCollection itself**
        and should return a new dict to merge with existing tags.

        ```python
        >>> c = collect(mono([1, 2], 44100), mono([3, 4], 44100))
        >>> c = c.map_collection_tags(lambda c: {"num_measurements": len(c.valid_entries)})
        >>> c.tags
        {'num_measurements': 2}

        ```
        """
        new_tags = fn(self)
        if not isinstance(new_tags, dict):
            raise TypeError(f"Function must return a dict, got {type(new_tags)}")
        return MonoCollection(self.entries, self.tags | new_tags)

    def filter(self, predicate: Callable[[Mono], bool]) -> "MonoCollection":
        """
        Filter the Mono entries based on a predicate function.

        ```python
        >>> x1 = mono([1, 2, 3], 44100)
        >>> x2 = mono([4, 5, 6], 44100)
        >>> xs = collect(x1, x2)
        >>> xs_filtered = xs.filter(lambda m: sum(m.data) > 6)
        >>> xs_filtered
        MonoCollection(
          entries=(
            Mono(data=[4, 5, 6], sample_rate=44100, tags={}),
          ),
          tags={}
        )

        ```
        """
        filtered_entries = [x for x in self.valid_entries if predicate(x)]
        return MonoCollection(tuple(filtered_entries), self.tags)

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
    masking_fn: Callable[[Mono], bool],
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
    - masking_fn (`Callable[[Mono], bool]`): A function that returns `True`
      for elements to include in the first output collection, and `False` for
      the second.

    ### Returns:

    A tuple of two `MonoCollection` object that preserve the structure of the
    it collection, with unmatched entries replaced by `None`.
    - The first contains elements where `masking_fn` returns `True`.
    - The second contains elements where it returns `False`."""

    filtered, rest = zip(
        *(
            (x, None) if x is None or masking_fn(x) else (None, x)
            for x in collection.entries
        )
    )
    return MonoCollection(tuple(filtered)), MonoCollection(tuple(rest))


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
    return MonoCollection(merged)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
