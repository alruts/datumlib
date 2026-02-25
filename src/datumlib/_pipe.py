from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import (
    Callable,
    Generic,
    Iterable,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)

from tqdm import tqdm

from datumlib._containers import Datum, DatumCollection, collect

D = TypeVar("D")


class Pipeline(Generic[D]):
    def __init__(self, steps: Iterable[tuple[str, Callable[[D], D]]]):
        self.steps: list[tuple[str, Callable[[D], D]]] = list(steps)

    @overload
    def run(
        self,
        x: Union[D, Datum[D], DatumCollection[D]],
        *,
        parallel: bool = ...,
        use_processes: bool = ...,
        max_workers: int | None = ...,
        trace: Literal[False] = False,
        progress_bar: bool = ...,
    ) -> Union[D, Datum[D], DatumCollection[D]]: ...

    @overload
    def run(
        self,
        x: Union[D, Datum[D], DatumCollection[D]],
        *,
        parallel: bool = ...,
        use_processes: bool = ...,
        max_workers: int | None = ...,
        trace: Literal[True],
        progress_bar: bool = ...,
    ) -> tuple[
        Union[D, Datum[D], DatumCollection[D]],
        dict[str, Union[D, Datum[D], DatumCollection[D]]],
    ]: ...

    def run(
        self,
        x: Union[D, Datum[D], DatumCollection[D]],
        *,
        parallel: bool = False,
        use_processes: bool = False,
        max_workers: int | None = None,
        trace: bool = False,
        progress_bar: bool = False,
    ) -> Union[
        D,
        Datum[D],
        DatumCollection[D],
        tuple[
            Union[D, Datum[D], DatumCollection],
            dict[str, Union[D, Datum[D], DatumCollection]],
        ],
    ]:
        trace_dict: dict[str, Union[D, Datum[D], DatumCollection]] = {}

        steps_iterable = (
            tqdm(self.steps, desc="Pipeline") if progress_bar else self.steps
        )

        for name, fn in steps_iterable:
            if isinstance(x, DatumCollection):
                entries: Iterable[D | None] = x.entries  # type: ignore

                def apply(entry: D | None) -> D | None:
                    return None if entry is None else fn(entry)

                if parallel:
                    Executor = (
                        ProcessPoolExecutor if use_processes else ThreadPoolExecutor
                    )
                    with Executor(max_workers=max_workers) as executor:
                        new_entries = list(executor.map(apply, entries))
                else:
                    new_entries = [apply(e) for e in entries]

                x = collect(*new_entries, tags=x.tags)  # type: ignore

            elif isinstance(x, Datum):
                # Datum contains a single entry
                x = cast(Datum[D], x)
                x = Datum(fn(x.data), tags=x.tags)

            else:
                # plain entry
                x = fn(cast(D, x))

            if trace:
                trace_dict[name] = x

        return (x, trace_dict) if trace else x
