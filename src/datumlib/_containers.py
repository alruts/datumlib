from logging import warning
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
)


class Datum(NamedTuple):
    """Data container for a 1D signal attached with relevant meta data. A `Datum`
    object is a NamedTuple which contains the following fields:

    - data (Any): The data to be stored in the container.
    - tags (dict): Meta data in the form of a dictionary. Default is an empty
    dictionary.

    ```python
    >>> x = Datum(1.0, {"some_number": 42})
    >>> x
    Datum(data=1.0, tags={'some_number': 42})
    >>> data, md = x
    >>> data, md
    (1.0, {'some_number': 42})
    >>> x.data # access via dot notation
    1.0
    >>> x.tags.get("some_number")
    42

    ```

    """

    data: Any
    tags: dict = {}

    @property
    def xp(self):
        """Return the array namespace for the underlying data."""
        if hasattr(self.data, "__array_namespace__"):
            return self.data.__array_namespace__()

        return None


class DatumCollection(NamedTuple):
    """Data container to batch together multiple Datum objects.

    A `DatumCollection` object is a NamedTuple which contains the following fields:
    - entries (`Tuple[Datum]`): A tuple of `Datum` objects.
    - tags (`dict`): Meta data in the form of a dictionary. Default is an empty
    dictionary.

    Datum collections can be constructed from multiple `Datum` objects using the
    `collect` function.

    >>> x = datum(1)
    >>> y = datum(1)
    >>> collect(x, y, tags={"date": "1999-12-31"})
    DatumCollection(
      entries=(
        Datum(data=1, tags={}),
        Datum(data=1, tags={}),
      ),
      tags={'date': '1999-12-31'}
    )

    """

    entries: tuple[Optional[Datum], ...]
    tags: dict = {}

    @property
    def valid_entries(self) -> list[Datum]:
        """Return a list of non-None Datum entries."""
        return [x for x in self.entries if x is not None]

    def __repr__(self) -> str:
        entry_lines = []
        for entry in self.entries:
            entry_lines.append(f"    {repr(entry)},")
        entries_repr = "\n".join(entry_lines)
        return (
            f"DatumCollection(\n"
            f"  entries=(\n{entries_repr}\n  ),\n"
            f"  tags={repr(self.tags)}\n"
            f")"
        )


# either
Container = Datum | DatumCollection


def datum(
    data: Any,
    tags: dict = {},
) -> Datum:
    """Create a datum signal container

    ```python
    >>> x = datum([1], {"level": 42})
    >>> x
    Datum(data=[1], tags={'level': 42})
    >>> data, md = x # unpacking
    >>> data, md
    ([1], {'level': 42})
    >>> x.data # access via dot notation
    [1]

    ```

    ### Args:
    - data (`Any`): The data to be stored in the container.
    - tags (`dict`): Meta data in the form of a dictionary. Default is an empty
    dictionary.

    ### Returns:
    A `NamedTuple` containing the data and meta data.
    - data (`Any`): The data to be stored in the
    container.
    - tags (`dict`): The meta data dictionary.

    """
    return Datum(data, tags)


def collect(*datums: Optional[Datum], tags: dict = {}) -> DatumCollection:
    """Collect multiple `Datum` objects into a `DatumCollection` with optional
    meta data.

    ```python
    >>> x = datum([1])
    >>> y = datum([1])
    >>> collect(x, y, tags={"session": "0000-00-00"})
    DatumCollection(
      entries=(
        Datum(data=[1], tags={}),
        Datum(data=[1], tags={}),
      ),
      tags={'session': '0000-00-00'}
    )

    ```

    ### Args:
    - datums (`Datum`): One or more datum objects.
    - tags (`dict`): Meta data in the form of a dictionary. Keyword only argument.

    ### Returns:
    A `NamedTuple` containing a tuple of `Datum` objects and the meta data.
    - entries (`Tuple[Datum]`): A tuple of `Datum` objects.
    - tags (`dict`): The meta data dictionary.
    """

    return DatumCollection(datums, tags)


def partition(
    collection: DatumCollection,
    masking_func: Callable[[Datum], bool],
) -> tuple[DatumCollection, DatumCollection]:
    """Splits `collection` into two separate `DatumCollection` objects based on
    a predicate function.

    Each `Datum` in the original collection is tested with `masking_fn`. If the
    function returns `True`, the `Datum` is placed in the first output
    collection; otherwise, it is placed in the second. The position of each
    `Datum` is preserved, and unmatched entries are replaced with `None`.

    ```python
    >>> x = datum([1], {"label": "a"})
    >>> y = datum([1], {"label": "b"})
    >>> xy = collect(x, y)
    >>> filtered, rest = partition(xy, lambda x: x.tags.get("label") == "a")
    >>> filtered
    DatumCollection(
      entries=(
        Datum(data=[1], tags={'label': 'a'}),
        None,
      ),
      tags={}
    )
    >>> rest
    DatumCollection(
      entries=(
        None,
        Datum(data=[1], tags={'label': 'b'}),
      ),
      tags={}
    )

    ```
    ### Args:
    - collection (`DatumCollection`): The it collection of `Datum` objects.
    - masking_func (`Callable[[Datum], bool]`): A function that returns `True`
      for elements to include in the first output collection, and `False` for
      the second.

    ### Returns:

    A tuple of two `DatumCollection` object that preserve the structure of the
    it collection, with unmatched entries replaced by `None`.
    - The first contains elements where `masking_fn` returns `True`.
    - The second contains elements where it returns `False`."""

    filtered, rest = zip(
        *(
            (x, None) if x is None or masking_func(x) else (None, x)
            for x in collection.entries
        )
    )
    return DatumCollection(tuple(filtered), collection.tags), DatumCollection(
        tuple(rest), collection.tags
    )


# helper for merge
def _check_and_pick(xs: tuple[Datum, ...]):
    hits = [x for x in xs if x is not None]
    if len(hits) > 1:
        raise ValueError("`DatumCollection` has overlapping (datum) entries.")
    return hits[0] if hits else None


def merge(*collections: DatumCollection) -> DatumCollection:
    """Merges `collections` objects (typically from `partition`) into a single
    DatumCollection.

    This function assumes that each collection shares the same structure, and
    that only one collection has a non-`None` entry at each position. This is
    typically used to reverse the effect of `partition`, after modifying the
    partitions separately.

    ```python
    >>> x = datum(1, {"label": "a"})
    >>> y = datum(1, {"label": "b"})
    >>> mc = collect(x, y)
    >>> filtered, rest = partition(mc, lambda x: x.tags.get("label") == "a")
    >>> merged = merge(
    ...     cmap(over_data(lambda x: x + 1))(filtered),
    ...     cmap(over_data(lambda x: x - 1))(rest)
    ... )  # apply different processing to each section
    >>> merged
    DatumCollection(
      entries=(
        Datum(data=2, tags={'label': 'a'}),
        Datum(data=0, tags={'label': 'b'}),
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

    tags = collections[0].tags  # !todo: this is sloppy
    return DatumCollection(merged, tags=tags)
