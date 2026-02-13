from typing import Callable, Generic, TypeVar

from tqdm import tqdm

from datumlib import Datum, DatumCollection

T = TypeVar("T", Datum, DatumCollection)


class PipelineDict(Generic[T]):
    def __init__(self, steps: dict[str, Callable[[T], T]]):
        self.steps = steps

    def __call__(
        self,
        x: T,
        *,
        has_scope: bool = False,
        progress_meter: bool = False,
    ) -> T | tuple[T, dict[str, T]]:
        scoped_results: dict[str, T] = {}

        # placeholder
        iter = (
            tqdm(
                self.steps.items(),
                desc=f"Processing {x.__class__.__name__}",
                unit="step",
            )
            if progress_meter
            else self.steps.items()
        )

        for name, func in iter:
            x = func(x)
            if has_scope:
                scoped_results[name] = x

        return (x, scoped_results) if has_scope else x


## todo: add something that takes in some params and returns a collection with
# different params (fx filterbank)
