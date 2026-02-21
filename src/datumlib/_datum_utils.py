from types import MappingProxyType
from typing import Any, Callable, Mapping

from datumlib import Datum, datum


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
        new_data = func(d.data)
        return datum(new_data, MappingProxyType(d.tags))

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
        new_data = func(d)
        return datum(new_data, d.tags)

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
        new_meta = func(d)
        if not isinstance(new_meta, Mapping):
            raise TypeError(
                f"Function passed to `update` must return a dictionary, got {type(new_meta)}."
            )
        return datum(d.data, dict(d.tags) | dict(new_meta))

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
        new_meta = func(d.tags)
        if not isinstance(new_meta, Mapping):
            raise TypeError(
                f"Function passed to `update` must return a mapping, got {type(new_meta)}."
            )
        return datum(d.data, dict(d.tags) | dict(new_meta))

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
    return datum(d.data, {**dict(d.tags), key: value})
