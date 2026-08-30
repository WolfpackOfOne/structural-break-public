"""The ONE shared streaming feature engine.

    stream --> StreamCtx (shared expensive state, updated ONCE per observation)
                  |
                  +--> StreamM00Core --.
                  +--> StreamM01Seq    |
                  +--> StreamM02Dist   +--> one concatenated feature vector
                  +--> StreamM03Dyn    |    (immutable column order)
                  +--> StreamM04Resid  |
                  +--> StreamM06Loc    |
                  +--> StreamM07Bayes -'
                                        |
                       every production booster reads a SLICE of it

Column order is fixed by `MODULE_ORDER` and, within a module, by that module's
own batch column order.  The manifest SHA over the full name list is what the
submission checks against the trained model.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from sbr.stream.ctx import StreamCtx
from sbr.stream.s_m00_core import StreamM00Core
from sbr.stream.s_m01_seq import StreamM01Seq
from sbr.stream.s_m02_dist import StreamM02Dist
from sbr.stream.s_m03_dyn import StreamM03Dyn
from sbr.stream.s_m04_resid import StreamM04Resid
from sbr.stream.s_m06_loc import StreamM06Loc
from sbr.stream.s_m07_bayes import StreamM07Bayes
from sbr.stream.s_m12_rdep import StreamM12Rdep

#: IMMUTABLE. Never reorder; APPEND ONLY. This is the order the batch pipeline
#: hstacks in.  `m12_rdep` is appended, not inserted: a model's manifest lists
#: the modules it was trained on and `StreamEngine` filters MODULE_ORDER by that
#: list, so appending cannot move a column of any existing model.  The RT-600
#: seven-module manifest SHA is asserted unchanged by
#: tests/test_stream_engine_parity.py::test_rt600_manifest_sha_is_unchanged.
MODULE_ORDER = ("m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid",
                "m06_loc", "m07_bayes", "m12_rdep")

#: The DEFAULT engine, and the shipped RT-600 architecture.  `MODULE_ORDER` is
#: the column ORDERING for everything the registry knows how to stream;
#: PRODUCTION_MODULES is what you get when you do not ask for anything specific.
#: A module is registered above so it can be requested by name and ordered
#: correctly, and it joins the default only when an artifact is actually built
#: on it -- otherwise `StreamEngine()` would silently start emitting columns no
#: shipped model was trained on.
PRODUCTION_MODULES = ("m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid",
                      "m06_loc", "m07_bayes")

_CLASSES = {
    "m00_core": StreamM00Core, "m01_seq": StreamM01Seq, "m02_dist": StreamM02Dist,
    "m03_dyn": StreamM03Dyn, "m04_resid": StreamM04Resid, "m06_loc": StreamM06Loc,
    "m07_bayes": StreamM07Bayes, "m12_rdep": StreamM12Rdep,
}


class StreamEngine:
    """Shared incremental engine over any subset of the production modules."""

    def __init__(self, modules=PRODUCTION_MODULES, ar_order: int = 2):
        bad = [m for m in modules if m not in _CLASSES]
        if bad:
            raise ValueError(f"unknown modules {bad}")
        self.modules = tuple(m for m in MODULE_ORDER if m in set(modules))
        self.ar_order = ar_order
        self._engines = [_CLASSES[m]() for m in self.modules]
        self.ctx = None
        self._cols = None
        self._widths = None

    # ------------------------------------------------------------------ fit
    def fit_historical(self, hist) -> StreamEngine:
        self.ctx = StreamCtx(ar_order=self.ar_order).fit_historical(
            np.asarray(hist, dtype=np.float64)
        )
        for e in self._engines:
            e.fit_historical(self.ctx)
        self._cols = []
        self._widths = []
        for m, e in zip(self.modules, self._engines):
            c = list(e.cols)
            self._widths.append(len(c))
            self._cols += [f"{m}::{x}" for x in c]
        return self

    # ----------------------------------------------------------------- step
    def step(self, x: float) -> np.ndarray:
        """Fold in one observation; return the full feature row (float32)."""
        self.ctx.push(x)
        parts = [e.step(self.ctx) for e in self._engines]
        row = np.concatenate(parts).astype(np.float32)
        row[~np.isfinite(row)] = np.nan
        return row

    # ---------------------------------------------------------------- meta
    @property
    def cols(self):
        if self._cols is None:
            raise RuntimeError("call fit_historical first")
        return self._cols

    @property
    def module_widths(self):
        return dict(zip(self.modules, self._widths))

    def manifest(self) -> dict:
        cols = self.cols
        return {
            "modules": list(self.modules),
            "n_features": len(cols),
            "columns": cols,
            "feature_manifest_sha256": hashlib.sha256("\n".join(cols).encode()).hexdigest(),
        }

    def save_manifest(self, path: str) -> str:
        m = self.manifest()
        json.dump(m, open(path, "w"), indent=1)
        return m["feature_manifest_sha256"]
