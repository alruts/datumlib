from typing import (
    Any,
    Callable,
)

from datumlib import Datum


def over_data(func: Callable[[Any], Any]) -> Callable[[Datum], Datum]:
    """Apply `fn` to the data field of the `Datum` container and return a transformed
    container

    ```python
    >>> x = datum(1)
    >>> over_data(lambda x: x + 1)(x)
    Datum(data=2, tags={})

    ```
    """

    def _over_data(d: Datum) -> Datum:
        new_data = func(d.data)
        return Datum(new_data, d.tags)

    return _over_data


def map_data(func: Callable[[Datum], Any]) -> Callable[[Datum], Datum]:
    """Apply `fn` to the whole `Datum` container and return a transformed
    container

    ```python
    >>> x = datum(1)
    >>> map_data(lambda x: x.data + 1)(x)
    Datum(data=2, tags={})

    ```
    """

    def _map_data(d: Datum) -> Datum:
        new_data = func(d)
        return Datum(new_data, d.tags)

    return _map_data


def map_tags(func: Callable[[Datum], dict]) -> Callable[[Datum], Datum]:
    """Update the metadata field of `datum` using `fn`.

    ```python
    >>> x = datum([1, 2, 3], {})
    >>> x = map_tags(lambda m: {"n_samples": len(m.data)})(x)
    >>> x.tags
    {'n_samples': 3}

    ```
    """

    def _map_tags(d: Datum) -> Datum:
        new_meta = func(d)
        if not isinstance(new_meta, dict):
            raise TypeError(
                f"Function passed to `update` must return a dictionary, got {type(new_meta)}."  # noqa: E501
            )
        return Datum(d.data, d.tags | new_meta)

    return _map_tags


def over_tags(func: Callable[[dict], dict]) -> Callable[[Datum], Datum]:
    """Update the metadata field of `datum` using `fn`.

    ```python
    >>> x = datum([1, 2, 3], {"idx": 3})
    >>> x = over_tags(lambda d: {"idx": d["idx"] + 1})(x)
    >>> x
    Datum(data=[1, 2, 3], tags={'idx': 4})

    ```
    """

    def _over_tags(d: Datum) -> Datum:
        new_meta = func(d.tags)
        if not isinstance(new_meta, dict):
            raise TypeError(
                f"Function passed to `update` must return a dictionary, got {type(new_meta)}."  # noqa: E501
            )
        return Datum(d.data, d.tags | new_meta)

    return _over_tags


def add_tags(d: Datum, key: str, value: Any) -> Datum:
    """Add tags to `datum`.

    ```python
    >>> x: Datum = datum([1, 2, 3], {})
    >>> x = add_tags(x, key="label", value="a")
    >>> x
    Datum(data=[1, 2, 3], tags={'label': 'a'})

    ```
    """
    return Datum(d.data, d.tags | {key: value})
