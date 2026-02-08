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


def dict_pipe(
    func_registry: dict[str, Callable[[Any], Any]], *, aux_result=False
) -> Callable[[Any], dict[str, Any] | tuple[dict[str, Any], Any]]:
    """Create left-to-right composition of functions with named results.

    ```python
    >>> pipeline = dict_pipe(
    ...     {
    ...         "after fst": lambda x: x + 1,
    ...         "after snd": lambda x: x + 2,
    ...         "result": lambda x: x / 2,
    ...     }
    ... )
    >>> pipeline(0)
    {'after fst': 1, 'after snd': 3, 'result': 1.5}

    ```

    Optionally set the `aux_result` kwd only flag to return the final output
    ```python
    >>> pipeline = dict_pipe(
    ...     {
    ...         "after fst": lambda x: x + 1,
    ...         "after snd": lambda x: x + 2,
    ...         "result": lambda x: x / 2,
    ...     },
    ...     aux_result=True,
    ... )
    >>> pipeline(0)
    ({'after fst': 1, 'after snd': 3, 'result': 1.5}, 1.5)

    ```
    """
    state = {}

    def _pipe(x):
        for name, f in func_registry.items():
            x = f(x)
            state.update({name: x})
        if aux_result:
            return state, x
        else:
            return state

    return _pipe


def compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Create right-to-left composition of functions.
    >>> compose(lambda x: x * 2, lambda x: x + 1)(1) # (1 + 1) * 2 = 4
    4
    """
    return lambda x: reduce(lambda acc, f: f(acc), reversed(funcs), x)


def tap(func: Callable[[X], Any]) -> Callable[[X], X]:
    """Runs a function with the input and returns the input unchanged
    (useful for debugging).

    >>> pipe(lambda x: x + 1, tap(print), lambda x: x + 1)(1)
    2
    3
    """
    return lambda x: (func(x), x)[1]


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
