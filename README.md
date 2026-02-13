# All of datumlib

## Container data types

The whole library is based on the `Datum` data type, which is simply a `NamedTuple` with three fields

- `data`: this field is reserved for a 1D array representing the values of a signal.
- `sample_rate`: the sample rate of said signal.
- `tags`: any relevant meta data that you wish to attach to said signal.

```{python}
from datumlib.containers import datum
from numpy import array

x = datum(array([1, 2, 3]), {"type": "B"})
x
```

Since `Datum` is a named tuple, we can deconstruct its components by either unpacking,
```{python}
data, tags = x
print(data, tags)
```
or by using its named attributes,
```{python}
print(x.data, x.tags)
```
`Datum` containers can be collected as `DatumCollection` types, which are nothing but a `NamedTuple` which includes a tuple of `Datum` containers along with optional `tags` for recording collection-wide meta data.

```{python}l
from datumlib.containers import collect

x1 = datum(array([1, 2, 3]), {"type": "A"})
x2 = datum(array([4, 5, 6]), {"type": "B"})
x3 = datum(array([7, 8, 9]), {"type": "C"})

collection = collect(x1, x2, x3, tags={"relevant": "info"})
collection
```

```{python}
type(collection.entries)
```

This abstraction allows us to define composable signal processing transformations which in a functional programming style, which allows us to do some cool and useful stuff.

## Chainable transformations


## Pretty printing
