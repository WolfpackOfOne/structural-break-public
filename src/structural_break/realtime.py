"""Streaming structural-break detection for the CrunchDAO "Real-Time Edition".

The Real-Time Edition of the ADIA Lab Structural Break Challenge differs from
the original challenge this package was first built around: instead of one
score per series at a known boundary point, each series has a break-free
*historical* segment followed by an *online* segment revealed one observation
at a time, with the break (if any) at an unknown point in that online segment.
A submission must yield exactly one score in ``[0, 1]`` per online
observation, in order, without ever revisiting a previous point.

This module implements an O(1)-per-point streaming detector plus the
``train``/``infer`` entry points required by ``crunch-cli``, and a small
synthetic-data / local-metric toolkit for development without the (licensed,
not redistributed) official data. See ``realtime_submission.ipynb`` for the
notebook used to actually submit to the competition.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import joblib
import numpy as np

__all__ = [
    "DetectorParams",
    "StreamingBreakDetector",
    "train",
    "infer",
    "time_stratified_auc",
    "make_realtime_series",
    "make_realtime_dataset",
]


@dataclass(frozen=True)
class DetectorParams:
    """Tunable hyperparameters for :class:`StreamingBreakDetector`.

    alpha -- EWMA decay in (0, 1]; higher reacts faster but is noisier.
    cusum_slack -- CUSUM slack (in historical standard deviations); larger
        ignores more noise before accumulating evidence.
    kappa_mean -- tanh scale for the mean-shift evidence.
    kappa_var -- tanh scale for the variance-shift evidence.
    """

    alpha: float = 0.05
    cusum_slack: float = 0.5
    kappa_mean: float = 3.0
    kappa_var: float = 1.5


_DEFAULT_PARAMS = DetectorParams()


class StreamingBreakDetector:
    """Streaming break-evidence score for a single time series.

    Summarizes the historical segment once at construction, then folds in
    online points one at a time via :meth:`update` in O(1) time and O(1)
    memory, combining three complementary signals:

    - an EWMA z-score of the running online mean against the historical mean
      (fast-reacting, decays old evidence);
    - a two-sided CUSUM of standardized residuals (accumulates evidence and
      never decays, so it also catches slow drifts the EWMA misses);
    - an EWMA variance-ratio test against the historical variance (catches
      volatility shifts that the two mean-based signals ignore).

    Each signal is squashed to ``[0, 1)`` with ``tanh`` and the three are
    combined with a noisy-OR (``1 - (1-a)(1-b)``), so any single strong
    signal drives the score toward 1 without requiring the others to agree.
    """

    def __init__(
        self,
        historical: Sequence[float],
        params: DetectorParams = _DEFAULT_PARAMS,
    ) -> None:
        x_h = np.asarray(historical, dtype=np.float64)
        self.mu_h = float(x_h.mean()) if len(x_h) else 0.0
        self.sd_h = float(x_h.std(ddof=1)) if len(x_h) > 1 else 1.0
        self.sd_h = max(self.sd_h, 1e-8)
        self.params = params

        self._mu_ewma = self.mu_h
        self._n_eff = 0.0
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._var_ewma = self.sd_h**2

    def update(self, x: float) -> float:
        """Fold in one new online observation; return the break score in [0, 1]."""
        p = self.params
        x = float(x)

        # --- mean-shift evidence: EWMA z-score of the running mean ---
        self._mu_ewma = (1.0 - p.alpha) * self._mu_ewma + p.alpha * x
        self._n_eff = (1.0 - p.alpha) * self._n_eff + 1.0
        se = self.sd_h / math.sqrt(max(self._n_eff, 1.0))
        z = (self._mu_ewma - self.mu_h) / max(se, 1e-8)
        ewma_evidence = math.tanh(abs(z) / p.kappa_mean)

        # --- mean-shift evidence: two-sided CUSUM of standardized residuals ---
        r = (x - self.mu_h) / self.sd_h
        self._cusum_pos = max(0.0, self._cusum_pos + r - p.cusum_slack)
        self._cusum_neg = min(0.0, self._cusum_neg + r + p.cusum_slack)
        cusum_stat = max(self._cusum_pos, -self._cusum_neg)
        cusum_evidence = math.tanh(cusum_stat / (2.0 * p.kappa_mean))

        mean_evidence = max(ewma_evidence, cusum_evidence)

        # --- variance-shift evidence: EWMA variance ratio vs. historical ---
        self._var_ewma = (1.0 - p.alpha) * self._var_ewma + p.alpha * (x - self.mu_h) ** 2
        ratio = self._var_ewma / (self.sd_h**2)
        var_evidence = math.tanh(abs(math.log(max(ratio, 1e-8))) / p.kappa_var)

        # --- combine: either signal alone can drive the score up ---
        return 1.0 - (1.0 - mean_evidence) * (1.0 - var_evidence)


def train(
    datasets: list[tuple[int, Sequence[float], Sequence[float], int | None]],
    model_directory_path: str,
    params: DetectorParams = _DEFAULT_PARAMS,
) -> None:
    """Required by crunch-cli.

    The detector has no learned parameters -- everything is computed from the
    historical segment and the streaming online values -- so this just
    persists the fixed hyperparameters that :func:`infer` will load.
    """
    del datasets  # unused: the baseline detector does not learn from labels
    joblib.dump(params, os.path.join(model_directory_path, "model.joblib"))


def infer(
    datasets: Iterable[tuple[Sequence[float], Iterable[float]]],
    model_directory_path: str,
) -> Iterator[float | None]:
    """Required by crunch-cli.

    A generator: yields once with no value to signal readiness, then exactly
    one ``float`` per online observation, in order, for every series in
    ``datasets``. Both ``datasets`` and each series' online iterable may only
    be consumed once.
    """
    params = joblib.load(os.path.join(model_directory_path, "model.joblib"))

    yield  # signal readiness to the runner

    for x_historical, x_online in datasets:
        detector = StreamingBreakDetector(x_historical, params)
        for point in x_online:
            yield detector.update(point)


def time_stratified_auc(
    scores: Sequence[Sequence[float]],
    labels: Sequence[Sequence[int]],
) -> float:
    """The competition's official metric, computed locally.

    ``scores[i]`` and ``labels[i]`` are the per-online-step score and
    ground-truth label (``1`` from the break onward, ``0`` before it or for
    the whole series if it has no break) for series ``i``, aligned by
    position within the online segment (not by absolute time, since series
    have different online lengths). At each step, compute the ordinary AUC
    across all series still "alive" at that step, then take a weighted
    average across steps (weight = number of positive/negative pairs).
    Returns ``0.5`` if no step ever has both classes present.
    """
    from sklearn.metrics import roc_auc_score

    max_len = max((len(s) for s in scores), default=0)
    weighted_sum = 0.0
    total_weight = 0.0
    for t in range(max_len):
        step_scores = [s[t] for s in scores if t < len(s)]
        step_labels = [y[t] for y in labels if t < len(y)]

        n_pos = sum(step_labels)
        n_neg = len(step_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        auc_t = float(roc_auc_score(step_labels, step_scores))
        weight = float(n_pos * n_neg)
        weighted_sum += weight * auc_t
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.5


def make_realtime_series(
    n_historical: int = 1200,
    n_online: int = 400,
    tau: int | None = None,
    mean_shift: float = 0.0,
    sd_ratio: float = 1.0,
    sd: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """One synthetic series in the challenge's ``(x_historical, x_online, tau)`` shape.

    If ``tau`` is ``None`` the online segment has no break: it is drawn from
    the same distribution as the historical segment throughout. Otherwise the
    first ``tau`` online points match the historical distribution and the
    remainder shift the mean by ``mean_shift`` and scale the standard
    deviation by ``sd_ratio``.
    """
    rng = np.random.default_rng(seed)
    x_historical = rng.normal(0.0, sd, n_historical)
    if tau is None:
        x_online = rng.normal(0.0, sd, n_online)
    else:
        pre = rng.normal(0.0, sd, tau)
        post = rng.normal(mean_shift, sd * sd_ratio, n_online - tau)
        x_online = np.concatenate([pre, post])
    return x_historical, x_online, tau


def make_realtime_dataset(
    n_series: int = 20,
    n_historical: int = 1000,
    n_online: int = 300,
    seed: int = 0,
) -> list[tuple[int, np.ndarray, np.ndarray, int | None]]:
    """A small mixed synthetic dataset in the challenge's tuple format.

    Returns a list of ``(dataset_id, x_historical, x_online, tau)`` tuples.
    Every other series gets a break (a mean shift, sometimes paired with a
    variance shift) at a random point in the online segment; the rest have no
    break at all. Deterministic for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    dataset = []
    for i in range(n_series):
        if i % 2 == 0:
            tau = int(rng.integers(20, n_online - 20))
            mean_shift = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(1.0, 2.5))
            sd_ratio = float(rng.uniform(1.2, 2.0)) if rng.random() < 0.5 else 1.0
        else:
            tau = None
            mean_shift = 0.0
            sd_ratio = 1.0

        x_h, x_o, tau = make_realtime_series(
            n_historical=n_historical,
            n_online=n_online,
            tau=tau,
            mean_shift=mean_shift,
            sd_ratio=sd_ratio,
            seed=seed * 1000 + i,
        )
        dataset.append((i, x_h, x_o, tau))
    return dataset
