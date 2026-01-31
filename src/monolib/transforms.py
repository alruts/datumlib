from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class PipelineDict(Generic[T]):
    """Dictionary-based pipeline runner with optional scoped output."""

    def __init__(self, steps: dict[str, Callable[[T], T]]):
        self.steps = steps

    def run(self, container: T, scope: bool = False) -> T | tuple[T, dict[str, T]]:
        """
        Run the pipeline sequentially.

        If `scope=True`, return a tuple of (final_result, scoped_results).
        If `scope=False`, return only the final result.
        """
        result = container
        scoped_results: dict[str, T] = {}

        for name, fn in self.steps.items():
            result = fn(result)
            if scope:
                scoped_results[name] = result

        if scope:
            return result, scoped_results
        return result

    def __call__(self, container: T, scope: bool = False) -> T | tuple[T, dict[str, T]]:
        return self.run(container, scope)
