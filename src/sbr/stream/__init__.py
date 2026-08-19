"""Incremental (streaming) feature engine.

ONE shared state object updates per observation; every production booster reads
a slice of the same feature vector.  See `sbr.stream.ctx.StreamCtx`.
"""
from sbr.stream.ctx import StreamCtx, TRANSFORM_NAMES  # noqa: F401
