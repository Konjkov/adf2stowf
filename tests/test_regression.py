"""Regression tests: convert every example TAPE21.asc and compare the produced
stowfn.data against the committed golden file in the same directory.

The goldens were generated with the default settings (``--cusp-method=project``,
occupied orbitals only).  Numeric tokens are compared with a tolerance so the
test guards against real changes in the conversion while tolerating last-bit
round-off (e.g. from an algebraically-equivalent refactor)."""

import contextlib
import io
import os
import pathlib
import shutil
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adf2stowf.adf2stowf import ADFToStoWF  # noqa: E402

EXAMPLES = REPO_ROOT / 'examples'

# Example directories that ship both the input and a golden output.
CASES = sorted(
    d.name
    for d in EXAMPLES.iterdir()
    if (d / 'TAPE21.asc').is_file() and (d / 'stowfn.data').is_file()
)


def _tokens(path):
    """Split a stowfn.data file into tokens, parsing numbers as floats."""
    tokens = []
    for tok in path.read_text().split():
        try:
            tokens.append(float(tok))
        except ValueError:
            tokens.append(tok)
    return tokens


@pytest.mark.parametrize('case', CASES)
def test_regression(case, tmp_path):
    src = EXAMPLES / case
    shutil.copy(src / 'TAPE21.asc', tmp_path / 'TAPE21.asc')

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ADFToStoWF(False, 'project', False, True).run()
    finally:
        os.chdir(cwd)

    got = _tokens(tmp_path / 'stowfn.data')
    want = _tokens(src / 'stowfn.data')

    assert len(got) == len(want), f'{case}: produced {len(got)} tokens, golden has {len(want)}'
    for i, (g, w) in enumerate(zip(got, want)):
        if isinstance(w, float):
            assert isinstance(g, float) and np.isclose(g, w, rtol=1e-8, atol=1e-10), f'{case}: token {i} differs: got {g!r}, want {w!r}'
        else:
            assert g == w, f'{case}: token {i} differs: got {g!r}, want {w!r}'
