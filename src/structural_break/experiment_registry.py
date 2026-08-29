"""Registry of experiments for the weekend model-search harness.

An *experiment* is a function ``fn(ctx: ExperimentContext, **config) ->
ExperimentResult`` registered with :func:`register`. The runner
(:mod:`structural_break.experiment_runner`) iterates :data:`EXPERIMENTS`, calls
each with its bound config, and records the resulting validation AUC to the
leaderboard.

To add a new experiment, decorate a function with ``@register(...)`` — see the
examples below. To sweep several hyperparameter combinations as separate
leaderboard rows, pass ``configs=[{...}, {...}, ...]``; each becomes its own
:class:`ExperimentSpec` with an id suffix describing its config.

Two kinds of entries live here:

1. **Working experiments** — ready to run today against either the synthetic
   competition dataset or real competition data.
2. **Stub experiments** (``hmm_regime``, ``bayesian_changepoint``,
   ``deep_learning``, ``stacking_ensemble``) — these currently raise
   ``NotImplementedError`` with a TODO describing the intended approach. The
   runner records them as "skipped", not a crash. Implement them to widen the
   search — that's the main way to extend this harness over the weekend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "ExperimentContext",
    "ExperimentResult",
    "ExperimentSpec",
    "EXPERIMENTS",
    "register",
]


@dataclass
class ExperimentContext:
    """Everything an experiment function needs, precomputed once per run.

    Feature matrices are computed once by the runner and shared across every
    experiment (instead of recomputed per-config) so a weekend-long sweep
    doesn't waste most of its time re-deriving the same features.
    """

    train_X: pd.DataFrame
    train_y: pd.Series
    val_X: pd.DataFrame
    val_y: pd.Series
    train_stat_features: pd.DataFrame
    val_stat_features: pd.DataFrame
    train_detector_features: pd.DataFrame
    val_detector_features: pd.DataFrame
    seed: int = 42
    artifact_dir: Path | None = None

    @property
    def train_combined_features(self) -> pd.DataFrame:
        return self.train_stat_features.join(self.train_detector_features)

    @property
    def val_combined_features(self) -> pd.DataFrame:
        return self.val_stat_features.join(self.val_detector_features)


@dataclass
class ExperimentResult:
    """What an experiment function returns."""

    auc: float
    model: Any | None = None
    notes: str = ""


@dataclass
class ExperimentSpec:
    """One registered, runnable (name, family, config) combination."""

    id: str
    family: str
    description: str
    fn: Callable[..., ExperimentResult]
    config: dict[str, Any] = field(default_factory=dict)


EXPERIMENTS: list[ExperimentSpec] = []


def register(
    name: str,
    family: str,
    description: str = "",
    configs: list[dict[str, Any]] | None = None,
) -> Callable[[Callable[..., ExperimentResult]], Callable[..., ExperimentResult]]:
    """Decorator: register ``fn`` under ``name``, once per entry in ``configs``."""

    def decorator(fn: Callable[..., ExperimentResult]) -> Callable[..., ExperimentResult]:
        for config in configs or [{}]:
            suffix = ""
            if config:
                suffix = "__" + ",".join(f"{k}={v}" for k, v in sorted(config.items()))
            EXPERIMENTS.append(
                ExperimentSpec(
                    id=f"{name}{suffix}",
                    family=family,
                    description=description,
                    fn=fn,
                    config=config,
                )
            )
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Working experiments
# ---------------------------------------------------------------------------


@register(
    "logistic_stat_features",
    family="linear_baseline",
    description="Logistic regression on before/after statistical features. Cheap floor.",
)
def logistic_stat_features(ctx: ExperimentContext, **_: Any) -> ExperimentResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=ctx.seed)),
        ]
    )
    model.fit(ctx.train_stat_features, ctx.train_y)
    proba = model.predict_proba(ctx.val_stat_features)[:, 1]
    auc = roc_auc_score(ctx.val_y, proba)
    return ExperimentResult(auc=auc, model=model)


@register(
    "rf_stat_features",
    family="ml_baseline",
    description="RandomForest on before/after statistical features.",
    configs=[
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 500, "max_depth": 8},
        {"n_estimators": 800, "max_depth": None},
    ],
)
def rf_stat_features(
    ctx: ExperimentContext, n_estimators: int = 200, max_depth: int | None = None
) -> ExperimentResult:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=ctx.seed
                ),
            ),
        ]
    )
    model.fit(ctx.train_stat_features, ctx.train_y)
    proba = model.predict_proba(ctx.val_stat_features)[:, 1]
    auc = roc_auc_score(ctx.val_y, proba)
    notes = f"n_estimators={n_estimators}, max_depth={max_depth}"
    return ExperimentResult(auc=auc, model=model, notes=notes)


@register(
    "rf_detector_ensemble",
    family="detector_ensemble",
    description="RandomForest on statistical features + CUSUM/rolling-z/PELT detector scores.",
)
def rf_detector_ensemble(ctx: ExperimentContext, **_: Any) -> ExperimentResult:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=500, random_state=ctx.seed)),
        ]
    )
    model.fit(ctx.train_combined_features, ctx.train_y)
    proba = model.predict_proba(ctx.val_combined_features)[:, 1]
    auc = roc_auc_score(ctx.val_y, proba)
    return ExperimentResult(auc=auc, model=model)


@register(
    "xgboost_combined_features",
    family="gradient_boosting",
    description="XGBoost on statistical + detector features.",
    configs=[
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.03},
    ],
)
def xgboost_combined_features(
    ctx: ExperimentContext,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
) -> ExperimentResult:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError("pip install xgboost to run this experiment.") from error

    from sklearn.metrics import roc_auc_score

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=ctx.seed,
        eval_metric="auc",
    )
    model.fit(ctx.train_combined_features, ctx.train_y)
    proba = model.predict_proba(ctx.val_combined_features)[:, 1]
    auc = roc_auc_score(ctx.val_y, proba)
    return ExperimentResult(auc=auc, model=model)


@register(
    "lightgbm_combined_features",
    family="gradient_boosting",
    description="LightGBM on statistical + detector features.",
)
def lightgbm_combined_features(ctx: ExperimentContext, **_: Any) -> ExperimentResult:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise ImportError("pip install lightgbm to run this experiment.") from error

    from sklearn.metrics import roc_auc_score

    model = LGBMClassifier(n_estimators=400, random_state=ctx.seed, verbosity=-1)
    model.fit(ctx.train_combined_features, ctx.train_y)
    proba = model.predict_proba(ctx.val_combined_features)[:, 1]
    auc = roc_auc_score(ctx.val_y, proba)
    return ExperimentResult(auc=auc, model=model)


# ---------------------------------------------------------------------------
# Stub experiments — extend these over the weekend.
# ---------------------------------------------------------------------------


@register(
    "hmm_regime",
    family="hmm_regime",
    description="Roadmap item: fit a 2-state HMM per series, score by the model's "
    "likelihood/posterior-regime disagreement between the before/after segments.",
)
def hmm_regime(ctx: ExperimentContext, **_: Any) -> ExperimentResult:
    # TODO: e.g. fit hmmlearn.hmm.GaussianHMM(n_components=2) per series on
    # `value`, then compare the dominant hidden state in period==0 vs period==1
    # (or the log-likelihood of a 1-state vs 2-state fit) as a break score.
    raise NotImplementedError(
        "hmm_regime is a stub — implement a per-series HMM regime-shift score "
        "(see the TODO in experiment_registry.py)."
    )


@register(
    "bayesian_changepoint",
    family="bayesian_changepoint",
    description="Roadmap item: online/offline Bayesian change-point detection "
    "(e.g. bayesian_changepoint_detection or a custom BOCPD) as a per-series score.",
)
def bayesian_changepoint(ctx: ExperimentContext, **_: Any) -> ExperimentResult:
    # TODO: run Bayesian Online Change Point Detection per series and use the
    # posterior run-length distribution's behaviour around the boundary index
    # as a break score/feature.
    raise NotImplementedError(
        "bayesian_changepoint is a stub — implement Bayesian change-point scoring "
        "(see the TODO in experiment_registry.py)."
    )


@register(
    "deep_learning_sequence_model",
    family="deep_learning",
    description="1D-CNN or LSTM classifier over the raw (padded/pooled) value "
    "sequence, predicting the per-series break label directly.",
)
def deep_learning_sequence_model(ctx: ExperimentContext, **_: Any) -> ExperimentResult:
    # TODO: e.g. a small 1D CNN (torch) over [value, period] per series, with
    # padding/masking for variable length, trained with BCE loss against
    # ctx.train_y. Requires torch — guard the import like the gradient-boosting
    # experiments above and raise ImportError with an install hint if missing.
    raise NotImplementedError(
        "deep_learning_sequence_model is a stub — implement a sequence model over "
        "the raw series (see the TODO in experiment_registry.py)."
    )


@register(
    "stacking_ensemble",
    family="stacking_ensemble",
    description="Meta-classifier stacking out-of-fold predictions from the other "
    "families (requires the runner to expose OOF predictions, not just val AUC).",
)
def stacking_ensemble(ctx: ExperimentContext, **_: Any) -> ExperimentResult:
    # TODO: once several base experiments are solid, generate out-of-fold
    # predictions for each (e.g. via cross_val_predict on train_combined_features)
    # and fit a meta-classifier (logistic regression is a good default) on top.
    raise NotImplementedError(
        "stacking_ensemble is a stub — implement OOF stacking across the other "
        "families (see the TODO in experiment_registry.py)."
    )
