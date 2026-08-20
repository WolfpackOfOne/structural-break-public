from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/scripts"))

from oracle_information_frontier import (  # noqa: E402
    MODEL_EXCLUDE_COLUMNS,
    PreparedSeries,
    assign_pseudo_taus,
    build_2026_records,
    current_model_scores,
    finite_auc,
    fit_ar_residuals,
    fold_separation_ok,
    model_feature_columns,
    one_row_per_series,
    parse_horizon,
    post_slice,
    pre_slice,
    score_models,
)


def test_horizon_and_true_tau_slicing() -> None:
    online = np.arange(10)
    assert parse_horizon("FULL") == "FULL"
    assert post_slice(online, 3, 2).tolist() == [3, 4]
    assert post_slice(online, 3, "FULL").tolist() == list(range(3, 10))
    assert post_slice(online, -1, 2).size == 0


def test_pre_window_and_full_post_extraction() -> None:
    online = np.arange(10)
    assert pre_slice(online, 6, 3).tolist() == [3, 4, 5]
    assert pre_slice(online, 6, "all").tolist() == [0, 1, 2, 3, 4, 5]
    assert post_slice(online, 9, "FULL").tolist() == [9]


def test_pseudo_tau_validity_and_determinism() -> None:
    meta = pd.DataFrame(
        {
            "id": np.arange(8),
            "fold": [0, 1, 2, 3, 4, 0, 1, 2],
            "target": [1, 1, 1, 1, 0, 0, 0, 0],
            "tau_index": [3, 5, 10, 20, -1, -1, -1, -1],
            "n_hist": [100] * 8,
            "n_online": [30, 30, 40, 50, 30, 35, 45, 12],
        }
    )
    a = assign_pseudo_taus(meta, 5, seed=7)
    b = assign_pseudo_taus(meta, 5, seed=7)
    assert a.equals(b)
    for r in meta[meta.target == 0].itertuples(index=False):
        tau = int(a.loc[int(r.id)])
        assert tau == -1 or tau + 5 <= int(r.n_online)


def test_model_feature_columns_exclude_boundary_metadata() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "fold": [0, 1],
            "target": [0, 1],
            "boundary": [3, 3],
            "rel_boundary": [0.3, 0.3],
            "n_hist": [100, 100],
            "n_online": [10, 10],
            "post_len": [5, 5],
            "hist__ks": [0.1, 0.9],
        }
    )
    cols = model_feature_columns(df)
    assert cols == ["hist__ks"]
    assert MODEL_EXCLUDE_COLUMNS.isdisjoint(cols)


def test_one_row_per_series_and_fold_separation() -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "fold": [0, 1, 1]})
    assert one_row_per_series(df)
    assert fold_separation_ok(df)
    bad = pd.DataFrame({"id": [1, 1], "fold": [0, 1]})
    assert not one_row_per_series(bad)
    assert not fold_separation_ok(bad)


def test_current_model_horizon_alignment(tmp_path: Path) -> None:
    meta = pd.DataFrame(
        {
            "id": [0, 1],
            "off": [0, 5],
            "n_hist": [10, 10],
            "n_online": [5, 4],
            "tau_index": [1, -1],
            "has_break": [1, 0],
        }
    )
    meta.to_parquet(tmp_path / "meta.parquet")
    preds = np.arange(9, dtype=np.float32)
    np.save(tmp_path / "RT.npy", preds)
    records = pd.DataFrame({"id": [0, 1], "boundary": [1, 2]})
    scores = current_model_scores(records, str(tmp_path / "meta.parquet"), str(tmp_path / "RT.npy"), 3)
    assert scores.tolist() == [3.0, 8.0]
    full = current_model_scores(records, str(tmp_path / "meta.parquet"), str(tmp_path / "RT.npy"), "FULL")
    assert full.tolist() == [4.0, 8.0]


def test_synthetic_mean_break_oracle_high_and_permutation_chance() -> None:
    rng = np.random.default_rng(0)
    series: dict[int, PreparedSeries] = {}
    for sid in range(120):
        target = int(sid < 60)
        fold = sid % 5
        hist = rng.normal(0, 1, 80)
        online = rng.normal(0, 1, 40)
        tau = 10 if target else -1
        if target:
            online[tau:] += 3.0
        hist_resid, online_resid = fit_ar_residuals(hist, online)
        series[sid] = PreparedSeries(
            id=sid,
            fold=fold,
            target=target,
            n_hist=len(hist),
            n_online=len(online),
            tau_index=tau,
            hist=hist,
            online=online,
            hist_resid=hist_resid,
            online_resid=online_resid,
        )
    df = build_2026_records(series, 20, pseudo_seed=0)
    metrics = score_models(df, seed=0, n_estimators=30)
    assert metrics["lgbm_rich"]["auc"] > 0.9

    shuffled = df.copy()
    shuffled["target"] = rng.permutation(shuffled["target"].to_numpy())
    perm_metrics = score_models(shuffled, seed=1, n_estimators=30)
    y_perm = shuffled["target"].to_numpy()
    auc = finite_auc(y_perm, perm_metrics["lgbm_rich"]["oof"])
    assert 0.3 <= auc <= 0.7
