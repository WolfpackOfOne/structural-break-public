"""The competition entry points: `train` and `infer`.

These are the functions the submission notebook exposes verbatim.  They obey
the Crunch Real-Time contract exactly (see research/reports/runner_semantics.md):

    infer(datasets, model_directory_path) is a GENERATOR.
    It yields once with no value to signal readiness, then exactly one score per
    online observation, in order, for every series in `datasets`.
    `datasets` and each series' online iterable are consumed ONCE.
    No cross-series information. No lookahead. `n_online` is never observed.
"""
from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np

from sbr.production.model import ProductionModel

#: The runner honours this. Series are independent, so parallelism is LEGAL --
#: and the fact that it is legal is itself the proof that no illegal
#: cross-series state exists.
#:
#: It is nevertheless 1, not 4. On the official macOS runner, P=4 segfaulted
#: LightGBM in two workers (`exit codes: 0=-11, 1=-11`) across three separate
#: attempts -- shared payload, process-local payload with a lazy LightGBM
#: import, and native thread caps -- see research/reports/crunch_test_rt150.md.
#: P=1 passed with the determinism check and projects 7.8 h on the private set
#: against a 15 h budget, so there is nothing to buy and a submission to lose.
#: research/scripts/build_submission.py already emits 1; this constant used to
#: disagree with the artifact that was actually tested.
INFER_PARALLELISM = 1


def train(datasets: list[tuple[int, list[float], list[float], int | None]],
          model_directory_path: str) -> None:
    """Fit and persist the model.

    A pre-built model directory (shipped with the submission) is honoured if it
    is already present and self-consistent -- retraining a 500-column, 600-tree
    booster over 10,000 series inside the runner's time budget is the single
    largest avoidable deployment risk, and the model is a frozen function of the
    training data either way.
    """
    if os.path.exists(os.path.join(model_directory_path, "manifest.json")):
        ProductionModel.load(model_directory_path)      # validates, raises if wrong
        return
    from sbr.production.fit import fit_from_datasets
    fit_from_datasets(datasets, model_directory_path)


def infer(datasets: Iterable[tuple[list[float], Iterable[float]]],
          model_directory_path: str):
    model = ProductionModel.load(model_directory_path)

    yield                                   # readiness handshake

    for x_historical, x_online in datasets:
        model.start_series(np.asarray(x_historical, dtype=np.float64))
        for point in x_online:
            yield model.step(float(point))
