from typing import (
    Callable,
    TypeVar,
)

from typing_extensions import Concatenate, ParamSpec

from datumlib import Datum, DatumCollection, collect

P = ParamSpec("P")
R = TypeVar("R", bound="Datum")


def filter_collection(
    c: DatumCollection, predicate: Callable[[Datum], bool]
) -> DatumCollection:
    """Filter a datum collection by a predicate"""
    return collect(*[x for x in c.valid_entries if predicate(x)], tags=c.tags)


def get_tags(c: DatumCollection, key: str, fill_with=None) -> list:
    """return all tags in a list"""
    return [x.tags.get(key, fill_with) for x in c.valid_entries]


def group_by_tag(c: DatumCollection, key: str) -> dict[str, DatumCollection]:
    all_tags = get_tags(c, key)
    unique_tags = list(set(all_tags))
    return {
        tag: filter_collection(c, lambda m: m.tags.get(key) == tag)
        for tag in unique_tags
    }


def compact(c: DatumCollection) -> DatumCollection:
    return collect(*c.valid_entries, tags=c.tags)


def cmap(
    func: Callable[Concatenate[Datum, P], R],
) -> Callable[Concatenate[DatumCollection, P], DatumCollection]:
    """Lift a datum transformation to operate over datum collections"""

    def _collection_map(
        collection: DatumCollection, *args: P.args, **kwargs: P.kwargs
    ) -> DatumCollection:
        return collect(
            *[
                func(x, *args, **kwargs) if x is not None else None
                for x in collection.entries
            ],
            tags=collection.tags,
        )

    return _collection_map
