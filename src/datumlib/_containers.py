from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    TypeVar,
)

T = TypeVar("T")


@dataclass(frozen=True)
class Datum(Generic[T]):
    """Immutable data container with metadata."""

    data: T
    tags: Mapping[str, object] = field(default_factory=dict, kw_only=True)

    def __post_init__(self):
        if not isinstance(self.tags, MappingProxyType):
            object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))

    def with_data(self, new_data: T):
        from datumlib._datum_utils import with_data

        return with_data(new_data)(self)

    def over_data(self, func: Callable[[T], T]):
        from datumlib._datum_utils import over_data

        return over_data(func)(self)

    def over_tags(self, func: Callable[[Mapping], Mapping]):
        from datumlib._datum_utils import over_tags

        return over_tags(func)(self)

    def map_data(self, func: Callable[["Datum[T]"], T]):
        from datumlib._datum_utils import map_data

        return map_data(func)(self)

    def map_tags(self, func: Callable[["Datum[T]"], Mapping]):
        from datumlib._datum_utils import map_tags

        return map_tags(func)(self)

    def add_tags(self, tags: Mapping):
        from datumlib._datum_utils import add_tags

        return add_tags(self, tags)


@dataclass(frozen=True)
class DatumCollection(Generic[T]):
    """Immutable collection of Datum objects."""

    entries: tuple[Optional[Datum[T]], ...]
    tags: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        if not isinstance(self.tags, MappingProxyType):
            object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))

    @property
    def valid_entries(self) -> tuple[Datum, ...]:
        return tuple(x for x in self.entries if x is not None)

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

    def with_data(self, new_data: T):
        from datumlib._datum_utils import with_data

        return cmap(with_data(new_data))(self)

    def over_data(self, func: Callable[[T], T]):
        from datumlib._datum_utils import over_data

        return cmap(over_data(func))(self)

    def over_tags(self, func: Callable[[Mapping], Mapping]):
        from datumlib._datum_utils import over_tags

        return cmap(over_tags(func))(self)

    def map_data(self, func: Callable[[Datum[T]], T]):
        from datumlib._datum_utils import map_data

        return cmap(map_data(func))(self)

    def map_tags(self, func: Callable[[Datum[T]], Mapping]):
        from datumlib._datum_utils import map_tags

        return cmap(map_tags(func))(self)

    def add_tags(self, key: str, values: list):
        return collect(
            *[
                x.add_tags({key: val}) if x else None
                for x, val in zip(self.entries, values)
            ],
            tags=self.tags,
        )


def cmap(
    func: Callable[[Datum], Datum], *, pass_tags: Optional[tuple[str]] = None
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

            kwds = {k: x.tags.get(k) for k in pass_tags} if pass_tags else {}
            mapped_entries.append(func(x, *args, **(kwargs | kwds)))

        return collect(*mapped_entries, tags=collection.tags)

    return _collection_map


def collect(
    *datums: Optional[Datum], tags: Mapping[str, object] | None = None
) -> DatumCollection:
    """Collects multiple `Datum` objects into a `DatumCollection`.

    ```python
    >>> x = datum(1, {"name": "bob"})
    >>> y = datum(2, {"name": "alice"})
    >>> z = datum(3, {"name": "john"})
    >>> collection = collect(x, y, z)
    >>> collection
    DatumCollection(
      entries=(
        Datum(data=1, tags=mappingproxy({'name': 'bob'})),
        Datum(data=2, tags=mappingproxy({'name': 'alice'})),
        Datum(data=3, tags=mappingproxy({'name': 'john'})),
      ),
      tags=mappingproxy({})
    )

    ```
    """
    return DatumCollection(
        entries=datums,
        tags=MappingProxyType(dict(tags)) if tags else MappingProxyType({}),
    )
