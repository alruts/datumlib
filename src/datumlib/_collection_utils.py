from functools import partial
from logging import warning
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
)

from datumlib._containers import Datum, DatumCollection, collect

T = TypeVar("T")
U = TypeVar("U")


def filter_collection(
    c: DatumCollection, predicate: Callable[[Datum], bool]
) -> DatumCollection:
    """Filter a datum collection by a predicate"""
    return collect(*[x for x in c.valid_entries if predicate(x)], tags=c.tags)


def get_tags(c: DatumCollection, key: str, fill_with: object | None = None) -> list:
    """todo: ..."""
    return [x.tags.get(key, fill_with) for x in c.valid_entries]


def group_by_tag(c: DatumCollection, key: str) -> dict[str, DatumCollection]:
    """todo: ..."""
    all_tags = get_tags(c, key)
    unique_tags = set(all_tags)
    return {
        tag: filter_collection(c, lambda m, k=key, t=tag: m.tags.get(k) == t)
        for tag in unique_tags
    }


def partition(
    collection: DatumCollection,
    masking_func: Callable[[Datum], bool],
) -> tuple[DatumCollection, DatumCollection]:
    """
    Splits collection into two separate `DatumCollection` objects based on a
    predicate function. Each Datum in the original collection is tested with
    `masking_fn`. If the function returns `True`, the `Datum` is placed in the
    first output collection; otherwise, it is placed in the second. The
    position of each `Datum` is preserved, and unmatched entries are replaced
    with `None`. For info about usefulness see `merge`.

    ```python
    >>> x = Datum(1, tags={"grade": "A"})
    >>> y = Datum(2, tags={"grade": "A"})
    >>> z = Datum(3, tags={"grade": "B"})
    >>> collection = collect(x, y, z)
    >>> As, Bs = partition(collection, lambda x: x.tags.get("grade") == "A")
    >>> As
    DatumCollection(
      entries=(
        Datum(data=1, tags=mappingproxy({'grade': 'A'})),
        Datum(data=2, tags=mappingproxy({'grade': 'A'})),
        None,
      ),
      tags=mappingproxy({})
    )

    >>> Bs
    DatumCollection(
      entries=(
        None,
        None,
        Datum(data=3, tags=mappingproxy({'grade': 'B'})),
      ),
      tags=mappingproxy({})
    )

    ```

    """
    filtered, rest = zip(
        *(
            (x, None) if x is None or masking_func(x) else (None, x)
            for x in collection.entries
        )
    )

    return (
        DatumCollection(filtered, collection.tags),
        DatumCollection(rest, collection.tags),
    )


def _check_and_pick(xs: tuple[Optional[Datum], ...]):
    hits = [x for x in xs if x is not None]
    if len(hits) > 1:
        raise ValueError("`DatumCollection` has overlapping (datum) entries.")
    return hits[0] if hits else None


def merge(*collections: DatumCollection) -> DatumCollection:
    """
    Merges masked `DatumCollection` objects (typically from `partition`) into a
    single `DatumCollection`. This function assumes that each collection shares
    the same structure, and that only one collection has a non-None entry at
    each position. This is typically used to reverse the effect of `partition`,
    after modifying the partitions separately.

    ```python
    >>> x = Datum(1, tags={"grade": "A"})
    >>> y = Datum(1, tags={"grade": "A"})
    >>> z = Datum(1, tags={"grade": "B"})
    >>> collection = collect(x, y, z, tags={"date": "1999-12-31"})
    >>> # Partition into 'A' and 'B' grades
    >>> As, Bs = partition(collection, lambda d: d.tags.get("grade") == "A")
    >>> # Suppose we increment data values in As
    >>> As_modified = As.over_data(lambda x: x + 1)
    >>> merged = merge(As_modified, Bs)
    >>> merged  # collection tags are preserved
    DatumCollection(
      entries=(
        Datum(data=2, tags=mappingproxy({'grade': 'A'})),
        Datum(data=2, tags=mappingproxy({'grade': 'A'})),
        Datum(data=1, tags=mappingproxy({'grade': 'B'})),
      ),
      tags=mappingproxy({'date': '1999-12-31'})
    )

    ```

    """

    entries = zip(*(c.entries for c in collections))
    merged = tuple(_check_and_pick(xs) for xs in entries)

    # Check tags consistency
    hashable_dicts = {tuple(sorted(c.tags.items())) for c in collections}

    if len(hashable_dicts) != 1:
        warning(
            "Tags for collections to be merged do not match! "
            "Using tags from the first collection."
        )

    return DatumCollection(entries=merged, tags=collections[0].tags)


def compact(c: DatumCollection) -> DatumCollection:
    """todo: ..."""
    return collect(*c.valid_entries, tags=c.tags)


def sort_by(
    c: DatumCollection,
    key: str,
    reverse: bool = False,
) -> DatumCollection:
    """
    Sort a DatumCollection.

    - `key`: if provided, sorts by the Datum's tag with this key.
    - `key_func`: optional custom function to extract a sort key from each Datum.
                  If both `key` and `key_func` are provided, `key_func` takes precedence.
    - `reverse`: if True, sorts in descending order.

    ```python
    >>> x = Datum(1, tags={"score": 10})
    >>> y = Datum(2, tags={"score": 30})
    >>> z = Datum(3, tags={"score": 20})
    >>> collection = collect(x, y, z)
    >>> sorted_collection = sort_by(collection, key="score")
    >>> [d.data for d in sorted_collection.valid_entries]
    [1, 3, 2]

    ```
    """

    def key_fn(entry):
        return entry.tags.get(key) if entry else ""

    sorted_entries = sorted(c.entries, key=key_fn, reverse=reverse)
    return DatumCollection(tuple(sorted_entries), tags=c.tags)


def zip_with(*collections: DatumCollection, func: Callable[[Datum], Any]) -> list[Any]:
    """Apply a transformation function across aligned entries of multiple
    DatumCollections, preserving positional correspondence."""
    return [func(*args) for args in zip(*map(lambda x: x.entries, collections))]


def compare_tags(*collections: DatumCollection, keys: list[str]) -> bool:
    """For each key, verify that all corresponding Datums across collections
    have identical tag values"""

    def _check_tags(*datums: Datum, key: str) -> bool:
        return len(datums) > 1 and len({d.tags.get(key) for d in datums}) == 1

    each_key = []
    for key in keys:
        all_vals_equal = all(zip_with(*collections, func=partial(_check_tags, key=key)))
        each_key.append(all_vals_equal)

    return all(each_key)
