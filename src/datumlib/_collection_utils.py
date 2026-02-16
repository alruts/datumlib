from typing import (
    Callable,
    Optional,
    TypeVar,
)

from datumlib import Datum, DatumCollection, collect

T = TypeVar("T")


def filter_collection(
    c: DatumCollection, predicate: Callable[[Datum], bool]
) -> DatumCollection:
    """Filter a datum collection by a predicate"""
    return collect(*[x for x in c.valid_entries if predicate(x)], tags=c.tags)


def get_tags(
    c: DatumCollection, key: str, fill_with: object | None = None
) -> list[object]:
    """return all tags in a list"""
    return [x.tags.get(key, fill_with) for x in c.valid_entries]


def group_by_tag(c: DatumCollection, key: str) -> dict[str, DatumCollection]:
    all_tags = get_tags(c, key)
    unique_tags: list[str] = [t for t in all_tags if isinstance(t, str)]
    return {
        tag: filter_collection(c, lambda m, k=key, t=tag: m.tags.get(k) == t)
        for tag in unique_tags
    }


def compact(c: DatumCollection) -> DatumCollection:
    return collect(*c.valid_entries, tags=c.tags)


def cmap(
    func: Callable[[Datum], Datum],
) -> Callable[[DatumCollection], DatumCollection]:
    """Lift a datum transformation to operate over datum collections."""

    def _collection_map(
        collection: DatumCollection, *args: object, **kwargs: object
    ) -> DatumCollection:
        mapped_entries: list[Optional[Datum]] = []
        for x in collection.entries:
            if x is None:
                mapped_entries.append(None)
                continue

            mapped_entries.append(func(x, *args, **kwargs))

        return collect(*mapped_entries, tags=collection.tags)

    return _collection_map
