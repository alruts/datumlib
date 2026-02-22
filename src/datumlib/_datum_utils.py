import dataclasses
from types import MappingProxyType
from typing import Any, Callable, Mapping

from datumlib import Datum, datum


def _get_class_and_fields(d: Datum):
    cls = type(d)
    fields = {
        f.name: getattr(d, f.name)
        for f in dataclasses.fields(d)
        if f.name not in ("data", "tags")
    }
    return cls, fields


def with_data(new_data: Any) -> Callable[[Datum], Datum]:
    """Apply `fn` to the data field of the `Datum` container and return a transformed
    container

    ```python
    >>> x = datum(1)
    >>> with_data(2)(x)
    Datum(data=2, tags=mappingproxy({}))

    ```
    """

    def _with_data(d: Datum) -> Datum:
        cls, fields = _get_class_and_fields(d)
        return cls(
            new_data,
            **fields,
            tags=d.tags,
        )

    return _with_data


def over_data(func: Callable[[Any], Any]) -> Callable[[Datum], Datum]:
    """Apply `fn` to the data field of the `Datum` container and return a transformed
    container

    ```python
    >>> x = datum(1)
    >>> over_data(lambda x: x + 1)(x)
    Datum(data=2, tags=mappingproxy({}))

    ```
    """

    def _over_data(d: Datum) -> Datum:
        cls, fields = _get_class_and_fields(d)
        new_data = func(d.data)
        return cls(
            new_data,
            **fields,
            tags=d.tags,
        )

    return _over_data


def map_data(func: Callable[[Datum], Any]) -> Callable[[Datum], Datum]:
    """Apply `fn` to the whole `Datum` container and return a transformed
    container

    ```python
    >>> x = datum(1)
    >>> map_data(lambda x: x.data + 1)(x)
    Datum(data=2, tags=mappingproxy({}))

    ```
    """

    def _map_data(d: Datum) -> Datum:
        cls, fields = _get_class_and_fields(d)
        new_data = func(d)
        return cls(
            new_data,
            **fields,
            tags=d.tags,
        )

    return _map_data


def map_tags(func: Callable[[Datum], Mapping[str, object]]) -> Callable[[Datum], Datum]:
    """Update the metadata field of `datum` using `fn`.

    ```python
    >>> x = datum([1, 2, 3], {})
    >>> x = map_tags(lambda m: {"n_samples": len(m.data)})(x)
    >>> x.tags
    mappingproxy({'n_samples': 3})

    ```
    """

    def _map_tags(d: Datum) -> Datum:
        cls, fields = _get_class_and_fields(d)
        new_meta = func(d)
        if not isinstance(new_meta, Mapping):
            raise TypeError(
                f"Function passed to `update` must return a dictionary, got {type(new_meta)}."
            )
        return cls(d.data, **fields, tags=dict(d.tags) | dict(new_meta))

    return _map_tags


def over_tags(
    func: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> Callable[[Datum], Datum]:
    """Update the metadata field of `datum` using `fn`.

    ```python
    >>> x = datum([1, 2, 3], {"idx": 3})
    >>> x = over_tags(lambda d: {"idx": d["idx"] + 1})(x)
    >>> x
    Datum(data=[1, 2, 3], tags=mappingproxy({'idx': 4}))

    ```
    """

    def _over_tags(d: Datum) -> Datum:
        cls, fields = _get_class_and_fields(d)
        new_meta = func(d.tags)
        if not isinstance(new_meta, Mapping):
            raise TypeError(
                f"Function passed to `update` must return a mapping, got {type(new_meta)}."
            )
        return cls(d.data, **fields, tags=dict(d.tags) | dict(new_meta))

    return _over_tags


def add_tags(d: Datum, key: str, value: object) -> Datum:
    """Add tags to `datum`.

    ```python
    >>> x: Datum = datum([1, 2, 3], {})
    >>> x = add_tags(x, key="label", value="a")
    >>> x
    Datum(data=[1, 2, 3], tags=mappingproxy({'label': 'a'}))

    ```
    """

    cls, fields = _get_class_and_fields(d)
    return cls(d.data, **fields, tags={**dict(d.tags), key: value})
