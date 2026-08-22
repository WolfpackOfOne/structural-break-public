#!/usr/bin/env python3
"""Fetch and VERIFY the 2026 feature store, then report what still has to be built.

The store is the only primary asset that has to move: 140,145,984 bytes of
`values.npy` plus a small `meta.parquet`. The ~10 GB feature cache is DERIVED and
is rebuilt locally afterwards.

Usage
-----
    # from a URL (S3/GCS presigned, GitHub release asset, any reachable host)
    python research/scripts/wave5_ingest_store.py --values-url URL --meta-url URL

    # from files already copied into the container
    python research/scripts/wave5_ingest_store.py --values-file P --meta-file P

    # verify an existing store without fetching anything
    python research/scripts/wave5_ingest_store.py --verify-only

Nothing is trusted. Both files are hashed and compared against the recorded
`store_values_sha256` / `store_meta_sha256` in
`research/REPRODUCIBILITY_MANIFEST.json`. A mismatch is a hard failure: a store
that is not byte-identical to the one the ledger was computed on would silently
re-base every number in this project.

`crunchdao.com` is blocked by this environment's network policy, so `crunch-cli`
cannot fetch the data here even though it installs. That is why this script takes
a URL on a reachable host instead of talking to the competition API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
STORE = os.environ.get("SBR_STORE", os.path.join(ROOT, "cache", "store"))
MANIFEST = os.path.join(ROOT, "research", "REPRODUCIBILITY_MANIFEST.json")

EXPECTED_VALUES_BYTES = 140_145_984


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def expected_hashes() -> tuple[str | None, str | None]:
    if not os.path.exists(MANIFEST):
        return None, None
    with open(MANIFEST) as fh:
        d = json.load(fh)
    data = d.get("data", {})
    return data.get("store_values_sha256"), data.get("store_meta_sha256")


def fetch(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  fetching {url[:80]}... -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as out:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            out.write(b)
    print(f"  wrote {os.path.getsize(dest):,} bytes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values-url")
    ap.add_argument("--meta-url")
    ap.add_argument("--values-file")
    ap.add_argument("--meta-file")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--store", default=STORE)
    a = ap.parse_args()

    vpath = os.path.join(a.store, "values.npy")
    mpath = os.path.join(a.store, "meta.parquet")

    if not a.verify_only:
        os.makedirs(a.store, exist_ok=True)
        if a.values_url:
            fetch(a.values_url, vpath)
        elif a.values_file:
            import shutil

            os.makedirs(a.store, exist_ok=True)
            shutil.copy2(a.values_file, vpath)
        if a.meta_url:
            fetch(a.meta_url, mpath)
        elif a.meta_file:
            import shutil

            shutil.copy2(a.meta_file, mpath)

    print(f"\nstore: {a.store}")
    missing = [p for p in (vpath, mpath) if not os.path.exists(p)]
    if missing:
        print("MISSING: " + ", ".join(missing))
        return 2

    exp_v, exp_m = expected_hashes()
    got_v = sha256_file(vpath)
    got_m = sha256_file(mpath)
    nbytes = os.path.getsize(vpath)

    print(f"  values.npy   {nbytes:,} bytes")
    print(f"    sha256 {got_v}")
    print(f"    expect {exp_v or '(not recorded)'}")
    print(f"  meta.parquet sha256 {got_m}")
    print(f"    expect {exp_m or '(not recorded)'}")

    ok = True
    if nbytes != EXPECTED_VALUES_BYTES:
        print(f"  !! size mismatch: expected {EXPECTED_VALUES_BYTES:,}")
        ok = False
    if exp_v and got_v != exp_v:
        print("  !! values.npy HASH MISMATCH -- refusing to certify this store")
        ok = False
    if exp_m and got_m != exp_m:
        print("  !! meta.parquet HASH MISMATCH -- refusing to certify this store")
        ok = False

    if not ok:
        print("\nSTORE NOT VERIFIED. Do not run experiments against it.")
        return 1

    print("\nSTORE VERIFIED byte-exact against the recorded manifest.")
    print("\nNext, in order:")
    print(f"  1. export SBR_STORE={a.store}")
    print(f"  2. export SBR_ROOT={ROOT}")
    print("  3. rebuild the feature cache (~7 min/module on 2 cores, ~10 GB):")
    print("       # the 7 frozen production modules, in the order the manifest indexes")
    print("       python -m sbr.features.driver --workers 4 --store $SBR_STORE \\")
    print("         --modules m00_core,m01_seq,m02_dist,m03_dyn,m04_resid,m06_loc,m07_bayes")
    print("       # then the wave-5 candidates, built separately so the control")
    print("       # arm can be run without them")
    print("       python -m sbr.features.driver --workers 4 --store $SBR_STORE \\")
    print("         --modules m10_perm,m11_focus")
    print("  4. confirm the folds are present: research/folds/folds.parquet")
    print("  5. run the pre-registered W5-E1 / W5-E2 controls in RDOF_LEDGER.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
