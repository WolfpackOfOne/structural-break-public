"""Model construction for the baseline structural-break classifier."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_random_forest_baseline(random_state: int = 42) -> Pipeline:
    """Build the baseline scikit-learn pipeline.

    The pipeline mirrors the original baseline: a :class:`~sklearn.preprocessing.StandardScaler`
    followed by a :class:`~sklearn.ensemble.RandomForestClassifier`. Scaling is not
    strictly necessary for a tree-based model, but it is kept here to preserve the
    original baseline behaviour.

    Parameters
    ----------
    random_state:
        Seed for the random forest, for reproducible results.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline ready to ``fit`` on engineered features.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=random_state)),
        ]
    )
