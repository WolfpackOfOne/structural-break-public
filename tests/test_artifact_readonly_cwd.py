"""The submission artifact must import from a read-only working directory.

LB-001 passed every local gate and then died on the cloud runner at *import*:

    File "/context/code/submissions/C_ensemble_deployable.py", line 21
      os.makedirs(_WORK, exist_ok=True)
    PermissionError: [Errno 13] Permission denied: '/context/code/_sbr_payload_96'

The boot cell unpacked its payload into `./_sbr_payload_<pid>` -- relative to the
current working directory. That is writable on a laptop and read-only at
`/context/code` on the runner, so no local `crunch test` could ever have caught
it: the local run just quietly littered the repo with the directory instead.

This test reproduces the runner's constraint directly, by importing the artifact
with cwd set to a directory the process cannot write to.
"""

import os
import subprocess
import sys
import tempfile

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(_REPO, "submissions", "C_ensemble_deployable.py")

needs_artifact = pytest.mark.skipif(
    not os.path.exists(ARTIFACT), reason=f"no built artifact at {ARTIFACT}")

_IMPORT_SNIPPET = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("artifact", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.INFER_PARALLELISM == 1, mod.INFER_PARALLELISM
assert callable(mod.train) and callable(mod.infer)
assert os.path.exists(os.path.join(mod.MODEL_DIRECTORY, "manifest.json")), \
    mod.MODEL_DIRECTORY
print("IMPORT_OK", mod.MODEL_DIRECTORY)
"""


def _import_artifact_with_cwd(cwd):
    return subprocess.run(
        [sys.executable, "-c", "import os\n" + _IMPORT_SNIPPET, ARTIFACT],
        cwd=cwd, capture_output=True, text=True, timeout=600)


@needs_artifact
def test_artifact_imports_with_a_read_only_cwd():
    with tempfile.TemporaryDirectory() as box:
        ro = os.path.join(box, "readonly")
        os.makedirs(ro)
        os.chmod(ro, 0o555)
        try:
            # root ignores the mode bits, which would make this test vacuous
            probe = os.path.join(ro, ".probe")
            try:
                open(probe, "w").close()
                os.remove(probe)
                pytest.skip("cwd is writable despite mode 0555 (running as root?)")
            except OSError:
                pass

            r = _import_artifact_with_cwd(ro)
            assert r.returncode == 0, (
                "the artifact cannot be imported from a read-only cwd -- this is "
                f"exactly the cloud-runner failure.\nSTDERR:\n{r.stderr}")
            assert "IMPORT_OK" in r.stdout, r.stdout
            assert not os.listdir(ro), (
                f"the artifact wrote into its cwd: {os.listdir(ro)}")
        finally:
            os.chmod(ro, 0o755)


@needs_artifact
def test_artifact_does_not_litter_a_writable_cwd():
    """Even where it *can* write to cwd, it should not."""
    with tempfile.TemporaryDirectory() as box:
        r = _import_artifact_with_cwd(box)
        assert r.returncode == 0, r.stderr
        assert not os.listdir(box), (
            f"the artifact created {os.listdir(box)} in its working directory; "
            "the payload belongs in a temp dir")


@needs_artifact
def test_payload_directory_is_honoured_when_set():
    """SBR_PAYLOAD_DIR gives an operator an explicit, writable choice."""
    with tempfile.TemporaryDirectory() as box:
        chosen = os.path.join(box, "chosen")
        env = dict(os.environ, SBR_PAYLOAD_DIR=chosen)
        r = subprocess.run(
            [sys.executable, "-c", "import os\n" + _IMPORT_SNIPPET, ARTIFACT],
            cwd=box, env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(chosen, "model", "manifest.json")), \
            os.listdir(chosen) if os.path.exists(chosen) else "not created"
