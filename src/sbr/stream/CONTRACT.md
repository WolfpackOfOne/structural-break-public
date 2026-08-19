# Streaming module contract (binding)

One shared `StreamCtx` (`sbr/stream/ctx.py`) holds ALL per-observation state that
is common to every module: historical params, the historical null-calibration
engine, the per-point transforms and their cumulative sums. It is **bitwise
identical** to the batch `make_ctx` (see `tests/test_stream_ctx_parity.py`).

Each feature module gets ONE streaming class in its OWN file
`src/sbr/stream/s_<module>.py`. Nobody edits anybody else's file, and nobody
edits `ctx.py` or anything under `src/sbr/features/`.

```python
class StreamM00Core:            # name: StreamM01Seq, StreamM02Dist, ...
    MODULE = "m00_core"

    def __init__(self): ...

    def fit_historical(self, ctx) -> None:
        """Called once, AFTER ctx.fit_historical(hist). Precompute anything that
        depends only on history. Reset all online state here."""

    def step(self, ctx) -> np.ndarray:
        """Called once per online observation, AFTER ctx.push(x_t).
        Return a float64 array of shape (k,) == row t of the batch module's
        output, in the SAME column order. Non-finite values must be np.nan
        (the batch modules do `A[~np.isfinite(A)] = np.nan` at the end -- you
        must reproduce that too)."""

    @property
    def cols(self) -> list[str]:
        """Column names, identical order to the batch module."""
```

## Rules
1. **Parity is the deliverable, not the features.** Target bitwise equality
   (`atol=0`, NaN==NaN) against `REGISTRY["<module>"].fn(make_ctx(hist, online))`
   row by row. Where a transform is numerically complicated and bitwise is
   genuinely unreachable, you must (a) prove it is unreachable, (b) report the
   max absolute and max relative deviation over the whole test set, (c) name the
   exact columns affected, and (d) justify the tolerance. "Close enough" without
   these four things is a failed deliverable.
2. **Cost.** Aim for O(1) or O(log w) work per observation; O(w) for a bounded
   window w (<= 512) is acceptable if the batch feature genuinely needs it.
   Forbidden: rescanning the whole online prefix at every step, re-sorting
   history, refitting AR models, recomputing an FFT over the full prefix.
   Report measured microseconds/observation for your module.
3. **No lookahead, ever.** `step` may read `ctx` state at the current index and
   your own carried state. It may not read `ctx.hist`-derived objects that were
   not available at fit time, and obviously not any future observation.
4. **The batch module is the specification.** If you think a batch feature is
   wrong, say so in your report; do not "fix" it. A streaming module that
   improves on batch is a parity failure.
5. Float64 internally; the caller casts to float32 once at the end.

## Useful `ctx` API at the current index t
- `ctx.t` (0-based), `ctx.n == t+1`
- `ctx.last(name)` transform value at t
- `ctx.roll(name, w)` trailing-window mean, NaN if fewer than w points
- `ctx.expand(name)` expanding mean over online[0..t]
- `ctx.cum[name]` full cumsum-with-leading-zero array (read `c[t+1]`, `c[t+1-w]`)
- `ctx.tr[name]` full per-point transform buffer (read backwards for windows)
- `ctx.window(name, w)` / `ctx.z_window(w)` last min(w, n) values
- `ctx.hp` HistParams, `ctx.nc` NullCal (`pct`, `z`, `surprise`, `grid`)

Note `ctx.roll`/`ctx.expand` return **scalars** here, where the batch versions
return whole arrays. That is the main mechanical difference when porting.

## Test file you must also write
`tests/test_stream_parity_<module>.py` containing:
- bitwise parity on >= 40 synthetic series spanning: short online (10-20),
  long online (900-999), constant history, near-zero-variance history,
  heavy-tailed (t3) history, strong AR(1) history, volatility-clustered,
  break-at-tau=0, break-at-the-last-point, and huge single outliers;
- bitwise parity on >= 30 random REAL series from `sbr.store.load_store()`;
- a prefix/poison test: build the stream, record the row at t, then confirm that
  mutating later observations cannot change it (structurally guaranteed, but
  test it);
- a timing test that prints microseconds per observation.
