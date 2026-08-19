"""The deployable model: shared streaming engine + one or more boosters.

Everything here must run under the Crunch streaming contract:
one series at a time, one observation at a time, no cross-series information,
no lookahead, no `n_online`.  See research/reports/runner_semantics.md.

A saved model directory contains:
    model.txt[.k]     LightGBM booster(s), text format (portable, version-stable)
    manifest.json     module list, ordered column names, feature-manifest SHA,
                      per-booster column slice, calibration payload, code SHA

`load()` fails loudly if the feature manifest the engine produces does not hash
to the value recorded at train time -- a model and a feature bank that disagree
is the failure mode that silently costs a competition, so it is a hard error.
"""
from __future__ import annotations

import json
import os

import numpy as np

from sbr.stream.engine import StreamEngine

MODEL_FILE = "model.txt"
MANIFEST_FILE = "manifest.json"


class ProductionModel:
    """Loaded, ready-to-stream predictor."""

    def __init__(self, boosters, manifest):
        self.boosters = boosters
        self.manifest = manifest
        self.modules = tuple(manifest["modules"])
        self.columns = list(manifest["columns"])
        self.slices = [np.asarray(s, dtype=np.int32) for s in manifest["booster_columns"]]
        self.calibration = manifest.get("calibration")
        self._cals = None
        if self.calibration and self.calibration.get("kind") == "scdf":
            from sbr.production.calibration import SmoothTimeCDFCal
            self._cals = [SmoothTimeCDFCal.from_json(x) for x in self.calibration["models"]]
        self._engine = None

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, model_directory_path: str) -> "ProductionModel":
        import lightgbm as lgb
        man = json.load(open(os.path.join(model_directory_path, MANIFEST_FILE)))
        boosters = []
        k = 0
        while True:
            p = os.path.join(model_directory_path, f"{MODEL_FILE}.{k}")
            if not os.path.exists(p):
                break
            boosters.append(lgb.Booster(model_file=p))
            k += 1
        if not boosters:
            boosters = [lgb.Booster(model_file=os.path.join(model_directory_path, MODEL_FILE))]
        m = cls(boosters, man)
        m._check_manifest()
        return m

    def _check_manifest(self):
        """Hard gate: the engine's columns must be exactly what was trained on."""
        eng = StreamEngine(self.modules)
        eng.fit_historical(np.arange(1200, dtype=np.float64) % 7 - 3.0)
        got = eng.manifest()
        if got["feature_manifest_sha256"] != self.manifest["feature_manifest_sha256"]:
            raise RuntimeError(
                "FEATURE MANIFEST MISMATCH -- the streaming engine does not produce the "
                "columns this model was trained on.\n"
                f"  model expects {self.manifest['feature_manifest_sha256']}\n"
                f"  engine yields {got['feature_manifest_sha256']}\n"
                "Refusing to run: predictions would be silently wrong.")

    # --------------------------------------------------------------- stream
    def start_series(self, historical) -> None:
        self._engine = StreamEngine(self.modules).fit_historical(historical)

    def step(self, x: float) -> float:
        row = self._engine.step(x)[None, :]
        t = self._engine.ctx.t
        # `validate_features=False, num_threads=1` is a 38x speedup per point
        # (1568 -> 41 us measured): the default path rebuilds a pandas-style
        # feature-name check and spins up a thread pool for a single row.
        # Verified to give bitwise-identical predictions.
        ps = [float(b.predict(row[:, s], validate_features=False, num_threads=1)[0])
              for b, s in zip(self.boosters, self.slices)]
        return self._blend(ps, t)

    # ------------------------------------------------------------ blending
    def _blend(self, ps, t: int) -> float:
        """Combine booster outputs into ONE score, using only per-series info."""
        cal = self.calibration
        if cal is None or len(ps) == 1:
            v = float(np.mean(ps))
        elif cal["kind"] == "logit_mean":
            z = [np.log(max(p, 1e-12) / max(1.0 - p, 1e-12)) for p in ps]
            v = 1.0 / (1.0 + np.exp(-float(np.mean(z))))
        elif cal["kind"] == "scdf":
            v = float(np.mean([c(p, t) for c, p in zip(self._cals, ps)]))
        else:
            raise ValueError(f"unknown calibration {cal['kind']!r}")
        if not np.isfinite(v):
            v = 0.5
        return float(min(max(v, 0.0), 1.0))
