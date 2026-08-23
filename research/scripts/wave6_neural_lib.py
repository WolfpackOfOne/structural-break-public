"""WAVE-6 NEURAL TRACK -- shared machinery.  Offline research only.

Binding pre-registration: research/WAVE6_NEURAL_PREREG.md, committed at 17d01b8
BEFORE any neural number existed.  Everything architectural in this file is a
transcription of that document.  Nothing here may be tuned after a score is
seen; if it changes, the experiment gets a new ID.

FORBIDDEN AND NOT REACHABLE FROM THIS FILE
    tau, future observations, n_online, final online length, boundary-
    conditioned availability, oracle features, the RT-500..RT-506 all-10k OOF
    vectors, folds_final10k.parquet, X_test.reduced / y_test.reduced, fold -1.

Tau appears in exactly two places in wave 6 and both are legal: constructing the
row labels y[t] = 1[t >= tau], which is the target, and post-hoc age-bucket
analysis.  It never enters a feature tensor, a normalisation constant, a mask, a
loss weight or a hidden state.
"""
from __future__ import annotations

import hashlib, json, os, platform, sys, time

import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

PROD_MODULES = ("m00_core", "m01_seq", "m02_dist", "m03_dyn",
                "m04_resid", "m06_loc", "m07_bayes")
#: torch CPU threads.  FIXED, because the reduction order of a multi-threaded
#: CPU kernel depends on the partition and the partition depends on this number.
TORCH_THREADS = 6


# --------------------------------------------------------------- determinism
def set_determinism(seed: int):
    """Gate 5.  CPU only -- MPS is deliberately not used, see the requirements file."""
    import torch
    os.environ.setdefault("PYTHONHASHSEED", "0")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_THREADS)
    torch.backends.cudnn.benchmark = False
    torch.backends.mkldnn.deterministic = True
    np.random.seed(seed)
    return torch.device("cpu")


def env_report() -> dict:
    import sklearn, scipy, pandas, lightgbm, torch
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(), "machine": platform.machine(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pandas.__version__, "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__, "torch": torch.__version__,
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_used": False,
        "torch_threads": TORCH_THREADS,
        "cpu_count": os.cpu_count(),
        "deterministic_algorithms": True,
    }


def sha_state_dict(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- row sampling
def champ_fold_rows(d, folds=(0, 1, 2, 3, 4), max_train_rows=1_000_000, seed=0):
    """The CHAMP protocol's training rows, reproduced EXACTLY.

    `sbr.pipeline.run` builds one rng before the fold loop and consumes it fold
    by fold, so the only way to hand the MLP literally the same rows RT-300 saw
    is to replay that sequence in the same order.  This is what makes track A an
    identical-information comparison rather than an approximate one.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for f in folds:
        tr_folds = [x for x in (0, 1, 2, 3, 4) if x != f]
        tr_rows = d.rows_for(tr_folds)
        if len(tr_rows) > max_train_rows:
            tr_rows = np.sort(rng.choice(tr_rows, max_train_rows, replace=False))
        out[f] = (tr_rows, d.rows_for([f]))
    return out


# ------------------------------------------------------- track A: the 500 cols
class FoldStandardiser:
    """Median/IQR from the TRAINING FOLD ONLY, frozen, then applied everywhere.

    Gate 4 (fold purity) and section 5 (normalisation is the red-alert area).
    Nothing here sees a validation row, a future observation, a sequence length
    or tau.  The constants are per-column scalars fitted on the sampled training
    rows of one fold and never refitted.
    """

    def __init__(self, group_sizes):
        self.med = None
        self.iqr = None
        self.group_sizes = tuple(group_sizes)

    def fit(self, X, col_block=40):
        # column-blocked: a float64 copy of a 1,000,000 x 500 matrix is 4 GB and
        # this machine has 16.  Quantiles are per column, so blocking is exact.
        n_col = X.shape[1]
        med = np.empty(n_col, dtype=np.float64)
        iqr = np.empty(n_col, dtype=np.float64)
        for a in range(0, n_col, col_block):
            b = min(a + col_block, n_col)
            q = np.nanquantile(np.asarray(X[:, a:b], dtype=np.float64),
                               [0.25, 0.5, 0.75], axis=0)
            med[a:b] = q[1]
            iqr[a:b] = q[2] - q[0]
        self.med = med.astype(np.float32)
        iqr = iqr.astype(np.float32)
        # a column that is constant on the training fold gets scale 1: dividing
        # by ~0 would manufacture enormous activations from float noise
        iqr[~np.isfinite(iqr) | (iqr <= 1e-12)] = 1.0
        self.med[~np.isfinite(self.med)] = 0.0
        self.iqr = iqr
        return self

    def transform(self, X, row_block=200_000):
        """(n, 500) -> (n, 500 + n_groups) float32.

        Row-blocked so peak memory stays flat; the map is row-wise, so blocking
        is exact.  Non-finite inputs become 0 AFTER standardising and raise the
        group's indicator channel, exactly as pre-registered.
        """
        n, k = X.shape
        g = len(self.group_sizes)
        out = np.empty((n, k + g), dtype=np.float32)
        for a in range(0, n, row_block):
            b = min(a + row_block, n)
            blk = np.asarray(X[a:b], dtype=np.float32)
            bad = ~np.isfinite(blk)
            Z = (blk - self.med) / self.iqr
            Z[bad] = 0.0
            np.clip(Z, -20.0, 20.0, out=Z)
            out[a:b, :k] = Z
            off = 0
            for j, w in enumerate(self.group_sizes):
                out[a:b, k + j] = bad[:, off:off + w].any(1)
                off += w
        return out


def build_mlp(n_in, dropout, seed, device):
    """500(+7) -> 256 -> 128 -> 32 -> 1, GELU, LayerNorm.  WAVE6_NEURAL_PREREG 2.1."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    layers, prev = [], n_in
    for w in (256, 128, 32):
        layers += [nn.Linear(prev, w), nn.LayerNorm(w), nn.GELU(), nn.Dropout(dropout)]
        prev = w
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers).to(device)


# ------------------------------------------------- track B: the legal channels
N_CHANNELS = 10
CHANNEL_NAMES = ("z", "z2", "absz", "sgn", "lag1", "run_mean", "run_var",
                 "ewma_fast", "ewma_slow", "elapsed")


def causal_channels(hist: np.ndarray, online: np.ndarray) -> np.ndarray:
    """The ten pre-registered channels.  (n_online, 10) float32.

    Row t uses `hist` and `online[:t+1]` and NOTHING ELSE.  `med`/`iqr` come from
    the historical segment, which is break-free by construction and complete
    before the online stream starts.  Every other channel is an expanding or
    exponentially-weighted running statistic.  `elapsed = log1p(t)/7` is an
    ELAPSED COUNTER known at time t; the final length never appears.
    """
    h = np.asarray(hist, dtype=np.float64)
    x = np.asarray(online, dtype=np.float64)
    n = len(x)
    med = float(np.median(h)) if len(h) else 0.0
    q75, q25 = (np.quantile(h, [0.75, 0.25]) if len(h) else (1.0, 0.0))
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr <= 1e-12:
        iqr = 1.0
    z = np.clip((x - med) / iqr, -8.0, 8.0)

    k = np.arange(1, n + 1, dtype=np.float64)
    cs = np.cumsum(z)
    cs2 = np.cumsum(z * z)
    run_mean = cs / k
    run_var = np.maximum(cs2 / k - run_mean ** 2, 0.0)

    def ewma(v, half_life):
        a = 1.0 - 0.5 ** (1.0 / half_life)
        out = np.empty(n)
        acc = 0.0
        for i in range(n):
            acc = acc + a * (v[i] - acc)
            out[i] = acc
        return out

    lag1 = np.empty(n)
    lag1[0] = 0.0
    if n > 1:
        lag1[1:] = z[1:] * z[:-1]

    C = np.empty((n, N_CHANNELS), dtype=np.float64)
    C[:, 0] = z
    C[:, 1] = z * z
    C[:, 2] = np.abs(z)
    C[:, 3] = np.sign(z)
    C[:, 4] = lag1
    C[:, 5] = run_mean
    C[:, 6] = np.log1p(run_var)
    C[:, 7] = ewma(z, 8.0)
    C[:, 8] = ewma(z, 64.0)
    C[:, 9] = np.log1p(np.arange(n, dtype=np.float64)) / 7.0
    C[~np.isfinite(C)] = 0.0
    np.clip(C, -20.0, 20.0, out=C)
    return C.astype(np.float32)


class CausalTCN:
    """Namespace holder so torch is imported lazily."""

    @staticmethod
    def build(hidden, dropout, seed, device, n_in=N_CHANNELS,
              kernel=3, dilations=(1, 2, 4, 8, 16, 32)):
        import torch
        from torch import nn
        torch.manual_seed(seed)

        class CausalConv(nn.Module):
            def __init__(self, cin, cout, d):
                super().__init__()
                self.pad = (kernel - 1) * d
                self.conv = nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(cin, cout, kernel, dilation=d))

            def forward(self, x):
                # LEFT pad only: the output at t sees t-pad .. t and never t+1.
                # Right padding would put future zeros inside the receptive field.
                return self.conv(nn.functional.pad(x, (self.pad, 0)))

        class Block(nn.Module):
            def __init__(self, cin, cout, d):
                super().__init__()
                self.c1 = CausalConv(cin, cout, d)
                self.c2 = CausalConv(cout, cout, d)
                self.act = nn.GELU()
                self.drop = nn.Dropout(dropout)
                self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

            def forward(self, x):
                h = self.drop(self.act(self.c1(x)))
                h = self.drop(self.act(self.c2(h)))
                return self.act(h + self.down(x))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [Block(n_in if i == 0 else hidden, hidden, d)
                     for i, d in enumerate(dilations)])
                self.head = nn.Conv1d(hidden, 1, 1)

            def forward(self, x):            # (B, C, T) -> (B, T)
                for b in self.blocks:
                    x = b(x)
                return self.head(x).squeeze(1)

        return Net().to(device)
