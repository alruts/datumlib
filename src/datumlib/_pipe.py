from typing import Callable, Collection, Mapping, Optional

from tqdm import tqdm

from datumlib._containers import Datum, DatumCollection, T, cmap

DatumContainer = Datum[T] | DatumCollection[T]


class Pipeline:
    def __init__(self, steps: Mapping[str, Callable[[Datum[T]], Datum[T]]]):
        self.steps = steps

    def __call__(
        self,
        x: DatumContainer,
        *,
        trace: bool = False,
        progress_bar: bool = False,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> DatumContainer | tuple[DatumContainer, Mapping[str, DatumContainer]]:
        trace_dict = {}
        steps_iter = tqdm(self.steps.items()) if progress_bar else self.steps.items()

        for label, func in steps_iter:
            if isinstance(x, DatumCollection):
                x = cmap(func, parallel=parallel, max_workers=max_workers)(x)
            elif isinstance(x, Collection):
                x = func(x)

            if trace:
                trace_dict[label] = x

        if trace:
            return (x, trace_dict)
        else:
            return x
