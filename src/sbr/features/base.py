"""Feature-module registry and the causal contract every module must honour.

A feature module is a function

    build(ctx) -> (list[str] names, np.ndarray (n_online, k) float32)

where ``ctx`` carries the per-series precomputed objects (historical params,
null-calibration engine, transforms).  The contract is:

    ROW t OF THE OUTPUT MAY DEPEND ONLY ON hist AND online[:t+1].

This is enforced empirically by ``check_prefix_invariance`` -- building on a
truncated online segment must reproduce the surviving rows bit for bit.  Any
module that fails is rejected; there is no "probably causal".
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from sbr.nullcal import NullCal
from sbr.transforms import HistParams, ar_filter_causal, build_transforms

REGISTRY: dict[str, FeatureModule] = {}


@dataclass
class FeatureModule:
    name: str
    version: str
    fn: Callable
    owner: str = ""
    doc: str = ""


def register(name: str, version: str = "1", owner: str = ""):
    def deco(fn):
        REGISTRY[name] = FeatureModule(
            name=name, version=version, fn=fn, owner=owner, doc=fn.__doc__ or ""
        )
        return fn
    return deco


def load_all():
    """Import every module in sbr.features so decorators populate the registry."""
    import sbr.features as pkg

    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name not in ("base", "driver"):
            importlib.import_module(f"sbr.features.{m.name}")
    return REGISTRY


@dataclass
class SeriesCtx:
    """Everything precomputed once per series and shared by all modules."""

    hist: np.ndarray
    online: np.ndarray
    hp: HistParams
    nc: NullCal
    tr: dict            # online transforms (per online point)
    cum: dict           # cumsum-with-leading-zero of each online transform
    n: int
    hist_tr: dict = field(default_factory=dict)
    ar_online: np.ndarray | None = None

    def roll(self, name: str, w: int) -> np.ndarray:
        """Trailing-window mean of transform ``name`` over window ``w``.

        Element t is the mean over online[t-w+1 .. t]; NaN while fewer than w
        online points have been seen (never back-filled from history, so the
        feature stays honest about how much online evidence exists).
        """
        c = self.cum[name]
        out = np.full(self.n, np.nan)
        if w <= self.n:
            out[w - 1:] = (c[w:] - c[:-w]) / w
        return out

    def expand(self, name: str) -> np.ndarray:
        """Expanding-window mean of transform ``name`` over online[0..t]."""
        c = self.cum[name]
        return c[1:] / np.arange(1, self.n + 1)


def make_ctx(
    hist: np.ndarray, online: np.ndarray, ar_order: int = 2, max_window: int | None = None
) -> SeriesCtx:
    hp = HistParams(hist, ar_order=ar_order)
    zh = (hist - hp.mu) / hp.sd
    from sbr.transforms import _ar_resid

    rh = np.concatenate([np.zeros(ar_order), _ar_resid(zh, hp.ar_coef)]) if ar_order else zh
    hist_tr = build_transforms(hist, hp, ar_resid=rh * hp.ar_sigma if ar_order else None)
    nc = NullCal(hist_tr, max_window=max_window)

    zo = (online - hp.mu) / hp.sd
    ar_on = ar_filter_causal(zo, hp.ar_coef, zh) * hp.ar_sigma if ar_order else None
    tr = build_transforms(online, hp, ar_resid=ar_on)
    cum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in tr.items()}
    return SeriesCtx(hist=hist, online=online, hp=hp, nc=nc, tr=tr, cum=cum,
                     n=len(online), hist_tr=hist_tr, ar_online=ar_on)


def check_prefix_invariance(module_name: str, hist: np.ndarray, online: np.ndarray,
                            cuts=(3, 10, 37), atol=0.0) -> tuple[bool, str]:
    """Rebuild the module on truncated online segments; rows must be identical."""
    mod = REGISTRY[module_name]
    names, full = mod.fn(make_ctx(hist, online))
    for k in cuts:
        if k >= len(online):
            continue
        _, part = mod.fn(make_ctx(hist, online[:k]))
        a = np.asarray(full[:k], dtype=np.float64)
        b = np.asarray(part, dtype=np.float64)
        bad = ~(np.isclose(a, b, rtol=0, atol=atol, equal_nan=True))
        if bad.any():
            j = int(np.argmax(bad.any(axis=0)))
            return False, f"{module_name}: prefix {k} differs, first bad column {names[j]}"
    return True, "ok"
