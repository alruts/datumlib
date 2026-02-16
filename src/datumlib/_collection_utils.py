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
    func: Callable[Concatenate["Datum", P], R], pass_tags: bool = False
) -> Callable[Concatenate["DatumCollection", P], "DatumCollection"]:
    """Lift a datum transformation to operate over datum collections.

    If `pass_tags` is True, each datum is passed `tags=collection.tags` as a keyword argument.
    """

    def _collection_map(
        collection: "DatumCollection", *args: P.args, **kwargs: P.kwargs
    ) -> "DatumCollection":
        mapped_entries = []
        for x in collection.entries:
            if x is None:
                mapped_entries.append(None)
                continue

            call_kwargs = dict(kwargs) | collection.tags
            mapped_entries.append(func(x, *args, **call_kwargs))

        return collect(*mapped_entries, tags=collection.tags)

    return _collection_map
