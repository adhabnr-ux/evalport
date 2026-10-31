import pytest

from evalport_wilson import wilson_interval

# Newcombe (1998), "Two-sided confidence intervals for the single proportion:
# comparison of seven methods", Statistics in Medicine 17:857-872, Table I,
# method 3 (score interval without continuity correction), 95%, 4 d.p.
NEWCOMBE_1998 = [
    (81, 263, 0.2553, 0.3662),
    (15, 148, 0.0624, 0.1605),
    (0, 20, 0.0000, 0.1611),
    (1, 29, 0.0061, 0.1718),
]

# Cross-checked against statsmodels 0.14.2,
# proportion_confint(successes, trials, alpha, method="wilson").
STATSMODELS = [
    (0, 10, 0.95, 0.000000, 0.277533),
    (10, 10, 0.95, 0.722467, 1.000000),
    (5, 10, 0.95, 0.236593, 0.763407),
    (1, 50, 0.95, 0.003539, 0.104954),
    (42, 50, 0.95, 0.714858, 0.916626),
    (42, 50, 0.99, 0.668518, 0.931819),
]


@pytest.mark.parametrize("s,n,low,high", NEWCOMBE_1998)
def test_matches_newcombe_1998(s, n, low, high):
    got = wilson_interval(s, n)
    assert got[0] == pytest.approx(low, abs=5e-5)
    assert got[1] == pytest.approx(high, abs=5e-5)


@pytest.mark.parametrize("s,n,level,low,high", STATSMODELS)
def test_matches_statsmodels(s, n, level, low, high):
    got = wilson_interval(s, n, level=level)
    assert got[0] == pytest.approx(low, abs=1e-6)
    assert got[1] == pytest.approx(high, abs=1e-6)


@pytest.mark.parametrize("n", [1, 2, 10, 50, 1000])
def test_zero_successes_keeps_an_upper_bound(n):
    low, high = wilson_interval(0, n)
    assert low == 0.0
    assert 0.0 < high < 1.0


@pytest.mark.parametrize("n", [1, 2, 10, 50, 1000])
def test_all_successes_keeps_a_lower_bound(n):
    low, high = wilson_interval(n, n)
    assert high == 1.0
    assert 0.0 < low < 1.0


@pytest.mark.parametrize("s,n", [(0, 7), (3, 7), (7, 7), (12, 50), (37, 50)])
def test_symmetric_in_successes_and_failures(s, n):
    low, high = wilson_interval(s, n)
    flip_low, flip_high = wilson_interval(n - s, n)
    assert low == pytest.approx(1.0 - flip_high, abs=1e-12)
    assert high == pytest.approx(1.0 - flip_low, abs=1e-12)


def test_contains_the_point_estimate():
    for s, n in [(1, 3), (17, 40), (199, 200)]:
        low, high = wilson_interval(s, n)
        assert low <= s / n <= high


def test_higher_level_is_wider():
    low95, high95 = wilson_interval(42, 50)
    low99, high99 = wilson_interval(42, 50, level=0.99)
    assert low99 < low95 and high99 > high95


def test_same_counts_same_interval():
    assert wilson_interval(13, 21) == wilson_interval(13, 21)


def test_accepts_integer_like_counts():
    class Count:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    assert wilson_interval(Count(5), Count(10)) == wilson_interval(5, 10)


@pytest.mark.parametrize(
    "s,n,level",
    [(0, 0, 0.95), (1, 0, 0.95), (-1, 10, 0.95), (11, 10, 0.95),
     (5, 10, 0.0), (5, 10, 1.0), (5, 10, 1.5), (5, 10, True)],
)
def test_rejects_invalid_values(s, n, level):
    with pytest.raises(ValueError):
        wilson_interval(s, n, level=level)


@pytest.mark.parametrize("s,n", [(5.0, 10), (5, 10.0), (True, 10), (1, True), ("5", 10)])
def test_rejects_non_integer_counts(s, n):
    with pytest.raises(TypeError):
        wilson_interval(s, n)
