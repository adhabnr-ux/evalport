"""Wilson score interval for a rate an adapter reports: successes out of trials.

Wilson is an episode-level interval. It assumes the trials are independent.
When they are correlated, for example several episodes or seeds of the same
task, pooling them as if they were independent gives an interval that is far
too narrow. Compute the interval over the unit that actually varies instead,
such as one success rate per task, or use a cluster-aware method.
"""

import math
import operator
from statistics import NormalDist
from typing import SupportsIndex, Tuple

__all__ = ["wilson_interval"]


def _count(name: str, value: SupportsIndex) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer count, not bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(
            f"{name} must be an integer count, not {type(value).__name__}"
        ) from None


def wilson_interval(
    successes: SupportsIndex, trials: SupportsIndex, *, level: float = 0.95
) -> Tuple[float, float]:
    """Return the two-sided Wilson score interval ``(low, high)``.

    Closed form, standard library only, and deterministic: the same counts and
    level always give the same interval. Unlike the normal-approximation
    (Wald) interval, it does not collapse to a point at 0/n or n/n: the low
    bound is exactly 0.0 at 0/n, the high bound exactly 1.0 at n/n, and the
    other bound stays informative.

    The trials must be independent. See the module docstring before using it
    on episodes that share a task or seed.

    Raises ``TypeError`` if a count is not an integer (booleans included), and
    ``ValueError`` if ``trials`` is below 1, ``successes`` is outside
    ``[0, trials]``, or ``level`` is not strictly between 0 and 1.
    """
    s = _count("successes", successes)
    n = _count("trials", trials)
    if n < 1:
        raise ValueError(f"trials must be at least 1, got {n}")
    if not 0 <= s <= n:
        raise ValueError(f"successes must be between 0 and trials ({n}), got {s}")
    if isinstance(level, bool) or not 0.0 < level < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level!r}")

    z = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    z2 = z * z
    p = s / n
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    low = 0.0 if s == 0 else max(0.0, center - half)
    high = 1.0 if s == n else min(1.0, center + half)
    return (low, high)
