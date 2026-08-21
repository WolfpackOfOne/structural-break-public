"""requirements.txt must cover everything inference actually imports.

The cloud runner builds its environment from requirements.txt. `crunch test`
does not: it runs in whatever venv you already have, with lightgbm and numba
sitting there from your research work. So a dependency can be missing from the
deployed environment while every local gate stays green.

That is how LB-002 died:

    File ".../sbr/production/model.py", line 48, in load
      import lightgbm as lgb
    ModuleNotFoundError: No module named 'lightgbm'

The dependency was never unknown -- requirements-research.txt listed lightgbm
and numba. It was declared in a file the runner does not read.

This test measures the import closure of the inference path in a subprocess and
asserts requirements.txt accounts for every third-party module in it.
"""

import json
import os
import re
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS = os.path.join(_REPO, "requirements.txt")
MODEL_DIR = os.environ.get(
    "SBR_MODEL_DIR", os.path.join(_REPO, "models", "final10k_ensemble"))

# Modules that ship inside another distribution or arrive as a transitive dep of
# one we do pin.  Anything NOT here must be named in requirements.txt directly.
_TRANSITIVE = {
    "llvmlite": "numba",
    "narwhals": "lightgbm",
    "dateutil": "pandas",
    "pytz": "pandas",
    "six": "pandas",
    "threadpoolctl": "scikit-learn",
    "joblib": "scikit-learn",
    "cffi": "pyarrow",
    "pycparser": "pyarrow",
    "cython_runtime": "numpy",
    "psutil": "numba",
}

# Loaded during inference but genuinely optional: each is a guarded import
# inside a package we DO declare, and its absence changes nothing.
#   charset_normalizer -- numpy/f2py/crackfortran.py:
#                         `try: import charset_normalizer / except ImportError: None`
#                         and f2py plays no part in inference.
#   yaml               -- numba/core/config.py: `try: import yaml` -> _HAVE_YAML,
#                         used only to read a file-based numba config.
# Verified against each package's Requires-Dist: numba needs only llvmlite and
# numpy; numpy declares no hard dependencies at all.
_OPTIONAL_GUARDED = {"charset_normalizer", "yaml"}

_PROBE = r'''
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["REPO"], "src"))
import numpy as np
from sbr.production.submission import infer
g = infer([(list(np.random.default_rng(0).standard_normal(600)),
            list(np.random.default_rng(1).standard_normal(40)))],
          os.environ["SBR_MODEL_DIR"])
next(g)
n = sum(1 for _ in g)
top = {k.split(".")[0] for k in sys.modules}
third = sorted(t for t in top
               if t not in sys.stdlib_module_names
               and not t.startswith("_")
               and t not in ("sbr",))
print("RESULT" + json.dumps({"scores": n, "third_party": third}))
'''


def _declared():
    """Distribution names named in requirements.txt, normalised."""
    names = set()
    for line in open(REQUIREMENTS, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)", line)
        if m:
            names.add(m.group(1).lower().replace("_", "-"))
    return names


needs_model = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, "manifest.json")),
    reason=f"no trained model at {MODEL_DIR}; set SBR_MODEL_DIR")


@needs_model
def test_requirements_cover_the_inference_import_closure():
    env = dict(os.environ, REPO=_REPO, SBR_MODEL_DIR=MODEL_DIR)
    r = subprocess.run([sys.executable, "-c", _PROBE],
                       env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout.split("RESULT", 1)[1].splitlines()[0])
    assert payload["scores"] == 40, payload

    declared = _declared()
    missing = []
    for mod in payload["third_party"]:
        if mod in _OPTIONAL_GUARDED:
            continue
        dist = _TRANSITIVE.get(mod, mod)
        dist = {"sklearn": "scikit-learn"}.get(dist, dist)
        if dist.lower().replace("_", "-") not in declared:
            missing.append(f"{mod} (needs `{dist}`)")
    assert not missing, (
        "inference imports these but requirements.txt does not declare them, so "
        "they will be absent on the cloud runner: " + ", ".join(missing))


def test_the_libraries_that_define_the_scored_function_are_pinned():
    """lightgbm IS the model and numba decides which numeric path runs."""
    text = open(REQUIREMENTS, encoding="utf-8").read()
    for pkg in ("lightgbm", "numba"):
        assert re.search(rf"^{pkg}==", text, re.M), (
            f"{pkg} must be pinned with ==, not left to float: it changes the "
            f"function that gets scored")


def test_runtime_requirements_are_not_weaker_than_research_requirements():
    """The deployed env must not be lighter than the one research validated on.

    requirements-research.txt listed lightgbm and numba while requirements.txt
    did not. That gap is exactly what shipped a broken submission.
    """
    research = os.path.join(_REPO, "requirements-research.txt")
    if not os.path.exists(research):
        pytest.skip("no requirements-research.txt")
    declared = _declared()
    gaps = []
    for line in open(research, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)", line)
        if m and m.group(1).lower().replace("_", "-") not in declared:
            gaps.append(m.group(1))
    assert not gaps, (
        "declared for research but not for the runner: " + ", ".join(gaps))
