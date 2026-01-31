from collections import namedtuple
from functools import reduce
from typing import Any, Callable, Iterable, List, NamedTuple, Tuple, TypeVar

X = TypeVar("X")  # Input type
Y = TypeVar("Y")  # Output type
Z = TypeVar("Z")  # Output type
Element = TypeVar("Element")  # Element type


def identity(x: X) -> X:
    """Returns the input unchanged.

    >>> identity(1)
    1
    """
    return x


def constant(x: X) -> Callable[[], X]:
    """Returns a function that always returns the input.

    >>> always_one = constant(1)
    >>> always_one()
    1
    """
    return lambda: x


def pipe(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Create right-to-left composition of functions.
    >>> pipe(lambda x: x * 2, lambda x: x + 1)(1) # (1 * 2) + 1 = 3
    3
    """
    return lambda x: reduce(lambda acc, f: f(acc), funcs, x)


def pipeline_dict(
    steps: dict[str, Callable[[Any], Any]], *, has_scope: bool = False
) -> Callable[[Any], Any | tuple[Any, dict[str, Any]]]:
    """Create left-to-right composition of functions with named results.

    Default returns only the final result (the last computed step):

    >>> pipeline = pipeline_dict(
    ...     {
    ...         "after fst": lambda x: x + 1,
    ...         "after snd": lambda x: x + 2,
    ...         "final": lambda x: x / 2,
    ...     }
    ... )
    >>> pipeline(0)
    1.5

    With has_scope=True, returns a tuple: (final result, dict of all steps):

    >>> pipeline = pipeline_dict(
    ...     {
    ...         "after fst": lambda x: x + 1,
    ...         "after snd": lambda x: x + 2,
    ...         "final": lambda x: x / 2,
    ...     },
    ...     has_scope=True,
    ... )
    >>> pipeline(0)
    (1.5, {'after fst': 1, 'after snd': 3, 'final': 1.5})
    """

    def _pipe(x):
        state = {}
        for name, f in steps.items():
            x = f(x)
            if has_scope:
                state[name] = x
        return (x, state) if has_scope else x

    return _pipe


def juxt(*funcs: Callable[[Any], Any]) -> Callable[[Any], Tuple[Any]]:
    """Applies multiple functions to the input and returns a tuple of the results.

    >>> juxt(lambda x: x, lambda x: x + 1)(1)
    (1, 2)
    """
    return lambda x: tuple(f(x) for f in funcs)


def aggregate_juxt(
    juxt_fn: Callable[[X], Tuple[Y, ...]], aggregator: Callable[[Y, Y], Any]
) -> Callable[[X], Any]:
    """Applies an aggregator to the results of a juxt to reduce them to a single value.

    Args:
        juxtaposition (Callable): A function that applies multiple functions to the input
        aggregator (Callable): A function that combines the results of the juxtaposition

    >>> aggregate_juxt(juxt(lambda x: x, lambda x: x), lambda x, y: x + y)(1)
    2
    """  # noqa: E501
    return lambda x: reduce(aggregator, (v for v in juxt_fn(x)))


def dict_juxt(
    fn_registry: dict[str, Callable[[X], Y]],
) -> Callable[[X], dict[str, Y]]:
    """Applies multiple functions to the input and returns a dictionary of the results.
    Each function is associated with a name.

    >>> dict_juxt({"a": lambda x: x, "b": lambda x: x})(1)
    {'a': 1, 'b': 1}
    """
    return lambda x: {name: func(x) for name, func in fn_registry.items()}


def zip_with(func: Callable[[Element], Y], *args: Iterable[Element]) -> List[Y]:
    """Pairs the elements of multiple lists and applies a function to them.

    >>> zip_with(lambda x, y: x + y, [1, 2], [3, 4])
    [4, 6]
    """
    return [func(*pair) for pair in zip(*args)]


def pair_named(name: str, **kwds: Iterable[Element]) -> List[NamedTuple]:
    """Pairs the elements of multiple iterables into a list of named tuples.

    >>> pair_named("point", x=[2, 2], y=[3, 4])
    [point(x=2, y=3), point(x=2, y=4)]
    """
    return [namedtuple(name, kwds.keys())(*pair) for pair in zip(*kwds.values())]


def pair(*args: Iterable[Element]) -> List[tuple]:
    """Pairs the elements of multiple iterables into a list of tuples.

    >>> pair([2, 2], [3, 4])
    [(2, 3), (2, 4)]
    """
    return [tuple(pair) for pair in zip(*args)]


def unpair(pairs: list[Tuple]) -> Tuple[List, ...]:
    """Pairs the elements of multiple iterables into a list of tuples.

    ```python
    >>> p = pair([2, 2], [3, 4])
    >>> p
    [(2, 3), (2, 4)]
    >>> unpair(p)
    ([2, 2], [3, 4])

    >>> p = pair_named("point", x=[2, 2], y=[3, 4])
    >>> p
    [point(x=2, y=3), point(x=2, y=4)]
    >>> unpair(p)
    ([2, 2], [3, 4])

    ```
    """
    return tuple(list(t) for t in zip(*pairs))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
