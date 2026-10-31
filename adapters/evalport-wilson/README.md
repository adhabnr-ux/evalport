# evalport-wilson

A Wilson score interval for the rates an adapter writes into a `ResultSet` entry: successes out of trials in, `(low, high)` out.

```python
from evalport_wilson import wilson_interval

wilson_interval(42, 50)              # (0.7148..., 0.9166...)
wilson_interval(0, 20)               # (0.0, 0.1611...), not (0.0, 0.0)
wilson_interval(42, 50, level=0.99)  # (0.6685..., 0.9318...)
```

- Closed form, standard library only (`statistics.NormalDist`), Python 3.8+.
- Deterministic: the same counts and level always give the same interval.
- At 0/n and n/n it does not collapse to a point the way the normal-approximation interval does.
- Rejects non-integer counts, `trials < 1`, `successes` outside `[0, trials]`, and a `level` outside (0, 1).

## When not to use it

Wilson is an episode-level interval and assumes independent trials. Episodes that share a task or a seed are correlated, and pooling them gives an interval that is far too narrow. Compute the interval over the unit that actually varies (one rate per task), or use a cluster-aware method.

## Tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

The tests check published values from Newcombe (1998), Statistics in Medicine 17:857-872, Table I, plus values cross-checked against statsmodels' `proportion_confint(method="wilson")`.
