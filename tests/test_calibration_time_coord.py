"""The SCDF time coordinate, and the t=0 case that motivated versioning it.

The competition's online index is zero-based.  The originally shipped calibrator
took logs of max(t, 1), which collapses t=0 onto t=1 and scores t=0 rows against
an anchor grid built from a window that excluded them.  Wave 4 (W4-E3) added
n_seen = t+1 as the corrected coordinate and made the payload record which one
it was fitted with.

The thing these tests actually protect: a payload fitted under one coordinate
and evaluated under the other is silently wrong -- no exception, no NaN, just a
differently-shaped monotone map.  The wave-2 artifact that passed `crunch test`
carries legacy payloads with no `time_coord` field at all.
"""
import numpy as np
import pytest

from sbr.production.calibration import (COORDS, DEFAULT_COORD, LEGACY_COORD,
                                        SmoothTimeCDFCal)


@pytest.fixture(scope="module")
def sample():
    """Shaped like the real store, which matters more than it looks.

    Online rows are series-major with ragged lengths, so LOW t is where the rows
    are: every series contributes a t=0 row, few reach t=800.  A uniform
    `randint(0, 900)` sample instead puts ~n/900 rows at each index, which drops
    every low anchor below min_n=400 and makes BOTH coordinates fall back to the
    pooled global grid -- hiding the very difference these tests exist to check.
    """
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 900, 2500)
    t = np.concatenate([np.arange(L) for L in lengths])
    s = rng.random(len(t)) * 0.9 + 0.05 * np.log1p(t) / 7
    return s, t


def test_legacy_collapses_t0_and_t1(sample):
    """Documents the bug, so nobody 'fixes' the legacy path and breaks the artifact."""
    cal = SmoothTimeCDFCal.fit(*sample, time_coord=LEGACY_COORD)
    assert cal(0.5, 0) == cal(0.5, 1)


def test_corrected_separates_t0_and_t1(sample):
    cal = SmoothTimeCDFCal.fit(*sample, time_coord=DEFAULT_COORD)
    assert cal(0.5, 0) != cal(0.5, 1)


def test_default_for_new_fits_is_corrected(sample):
    assert SmoothTimeCDFCal.fit(*sample).time_coord == DEFAULT_COORD


def test_payload_without_the_field_is_treated_as_legacy(sample):
    """Every artifact built before wave 4 -- including the Crunch-tested one."""
    d = SmoothTimeCDFCal.fit(*sample, time_coord=LEGACY_COORD).to_json()
    del d["time_coord"]
    assert SmoothTimeCDFCal.from_json(d).time_coord == LEGACY_COORD


@pytest.mark.parametrize("coord", COORDS)
def test_round_trip_preserves_coordinate_and_values(sample, coord):
    s, t = sample
    a = SmoothTimeCDFCal.fit(s, t, time_coord=coord)
    b = SmoothTimeCDFCal.from_json(a.to_json())
    assert b.time_coord == coord
    for tt in (0, 1, 2, 5, 100, 899):
        assert a(0.5, tt) == b(0.5, tt)


@pytest.mark.parametrize("coord", COORDS)
def test_output_is_a_probability_everywhere(sample, coord):
    s, t = sample
    cal = SmoothTimeCDFCal.fit(s, t, time_coord=coord)
    for tt in (0, 1, 3, 47, 899, 5000):          # 5000 is beyond the top anchor
        for x in (-1e9, 0.0, 0.5, 1.0, 1e9, float("inf")):
            v = cal(x, tt)
            assert 0.0 <= v <= 1.0 and np.isfinite(v)


@pytest.mark.parametrize("coord", COORDS)
def test_monotone_in_score(sample, coord):
    s, t = sample
    cal = SmoothTimeCDFCal.fit(s, t, time_coord=coord)
    for tt in (0, 7, 200):
        v = [cal(x, tt) for x in np.linspace(0.0, 1.0, 50)]
        assert all(b >= a for a, b in zip(v, v[1:]))


def test_unknown_coordinate_is_rejected(sample):
    with pytest.raises(ValueError):
        SmoothTimeCDFCal.fit(*sample, time_coord="log_t_squared")
    with pytest.raises(ValueError):
        SmoothTimeCDFCal([1, 2], [np.array([0.0]), np.array([0.0])], time_coord="nope")


def test_legacy_excludes_t0_from_the_grid_it_scores_t0_against():
    """The mechanism, not the symptom.

    Under the corrected coordinate a t=0 row sits at n_seen=1, which IS the first
    anchor, so the grid it is scored against was built from rows like it.  Under
    the legacy coordinate the fit cuts windows on RAW t: t=0 falls below the
    first window's lower edge (a*exp(-half) ~= 0.54) and is excluded, while the
    evaluation clamps to log(1) and routes it to that same anchor anyway.
    """
    # every series reaches t=899, so each index has 600 rows -- comfortably above
    # min_n, which keeps the fallback path out of the comparison
    t = np.repeat(np.arange(0, 900), 600)
    rng = np.random.default_rng(1)
    # t=0 scores are disjoint from every other index, so inclusion is visible
    s = np.where(t == 0, 0.99, rng.random(len(t)) * 0.5)

    corrected = SmoothTimeCDFCal.fit(s, t, time_coord=DEFAULT_COORD)
    legacy = SmoothTimeCDFCal.fit(s, t, time_coord=LEGACY_COORD)

    assert corrected.grids[0].max() > 0.9, "t=0 rows should build the grid that scores them"
    assert legacy.grids[0].max() < 0.9, "legacy should exclude them -- that is the bug"
    # and the consequence: legacy maps a typical t=0 score to the top of a grid
    # made entirely of other timesteps
    assert legacy(0.99, 0) == 1.0
    assert corrected(0.99, 0) < 1.0
