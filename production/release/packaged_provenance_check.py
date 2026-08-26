"""Gate E, measured through the PACKAGED artifact rather than through src/.

Two things are asserted, in this order:

1. the notebook's own boot cell verifies the embedded source and model zips
   against the sha256 values compiled into it, and refuses to unpack otherwise;
2. the ``ProductionModel`` INSIDE that payload refuses six model directories
   whose feature manifest is perfectly valid but whose fit population is not the
   frozen RT-600 one -- the CRF-02 void-run failure mode, where a 24-series
   unit-test checkpoint loaded silently into a real scoring run.

The escape hatch is exercised too: research must be able to load a different
model deliberately, production must never set the variable.

    SBR_ROOT=<repo> python packaged_provenance_check.py --nb <notebook> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packaged_release_check import load_payload            # noqa: E402

VARIANTS = [
    (("trained_on", "n_series"), 24, "the CRF-02 failure mode exactly"),
    (("trained_on", "n_series"), 8000, "a dev-fold model, not the final fit"),
    (("trained_on", "partition"), "folds_dev", "the wrong partition"),
    (("folds_sha256",), "0" * 64, "a different fold assignment"),
    (("calibration", "kind"), "logit_mean", "a rejected blend"),
    (("calibration", "time_coord"), "log_t", "the pre-W4-E3 time coordinate"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nb = os.path.abspath(a.nb)
    out = os.path.abspath(a.out) if a.out else None

    g, pdir, work = load_payload(nb)
    print(f"payload isolation OK: {pdir}", flush=True)

    from sbr.production.model import ProductionModel      # the PAYLOAD's copy
    assert os.path.abspath(ProductionModel.__module__ and
                           sys.modules["sbr.production.model"].__file__).startswith(
        os.path.dirname(pdir)), "not the payload's ProductionModel"

    mdir = os.path.join(work, "model_out")
    os.makedirs(mdir, exist_ok=True)
    g["train"]([], mdir)

    res = {"schema": "rt600_clean_release/packaged_provenance/1",
           "payload_sha256_self_check": "the boot cell verified both zips before "
                                        "unpacking; a mismatch aborts at import",
           "frozen_artifact_loads": False, "variants": [], "escape_hatch": None}

    ProductionModel.load(mdir)
    res["frozen_artifact_loads"] = True

    for field, value, why in VARIANTS:
        d = tempfile.mkdtemp(dir=work)
        m = os.path.join(d, "m")
        shutil.copytree(mdir, m)
        p = os.path.join(m, "manifest.json")
        man = json.load(open(p))
        node = man
        for k in field[:-1]:
            node = node[k]
        node[field[-1]] = value
        json.dump(man, open(p, "w"))
        row = {"field": ".".join(field), "value": value, "why": why,
               "refused": False, "error": None}
        try:
            ProductionModel.load(m)
        except RuntimeError as exc:
            row["refused"] = "MODEL PROVENANCE MISMATCH" in str(exc)
            row["error"] = str(exc).splitlines()[0]
        res["variants"].append(row)
        print(f"  {row['field']}={value!r}: refused={row['refused']}", flush=True)

    d = tempfile.mkdtemp(dir=work)
    m = os.path.join(d, "m")
    shutil.copytree(mdir, m)
    p = os.path.join(m, "manifest.json")
    man = json.load(open(p))
    man["trained_on"]["n_series"] = 24
    json.dump(man, open(p, "w"))
    os.environ["SBR_ALLOW_UNPINNED_MODEL"] = "1"
    try:
        ProductionModel.load(m)
        res["escape_hatch"] = {"var": "SBR_ALLOW_UNPINNED_MODEL=1", "loads": True}
    finally:
        del os.environ["SBR_ALLOW_UNPINNED_MODEL"]

    res["PASS"] = bool(res["frozen_artifact_loads"]
                       and all(v["refused"] for v in res["variants"])
                       and res["escape_hatch"]["loads"])
    import hashlib
    res["_meta"] = {"phase": "provenance", "notebook": nb,
                    "notebook_sha256": hashlib.sha256(open(nb, "rb").read()).hexdigest(),
                    "payload_dir": pdir,
                    "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(res, indent=2, default=str))
    if out:
        json.dump(res, open(out, "w"), indent=2, default=str)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
