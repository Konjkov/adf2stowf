"""Regression tests: compare adf2stowf output against reference stowfn.data files.

Each test runs the full conversion pipeline on examples/<system>/TAPE21.asc
and compares the resulting stowfn.data against the reference file in the same
directory, using the project's own StoWfn class for parsing.

Run with::

    pytest tests/test_regression.py -v

or for a single system::

    pytest tests/test_regression.py -k He -v
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / 'examples'
sys.path.insert(0, str(REPO_ROOT))

from adf2stowf.adf2stowf import ADFToStoWF  # noqa: E402
from adf2stowf.stowfn import StoWfn         # noqa: E402

# Systems that have both TAPE21.asc and a reference stowfn.data
SYSTEMS = sorted(
    d.name
    for d in EXAMPLES_DIR.iterdir()
    if d.is_dir()
    and (d / 'TAPE21.asc').exists()
    and (d / 'stowfn.data').exists()
)


def _convert(system: str) -> StoWfn:
    """Run the converter in a temp dir and return the parsed StoWfn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(EXAMPLES_DIR / system / 'TAPE21.asc', Path(tmpdir) / 'TAPE21.asc')
        orig = os.getcwd()
        try:
            os.chdir(tmpdir)
            ADFToStoWF(
                plot_cusps=False,
                cusp_method='enforce',
                do_dump=False,
                cart2harm_projection=False,
                only_occupied=True,
            ).run()
            return StoWfn(str(Path(tmpdir) / 'stowfn.data'))
        finally:
            os.chdir(orig)


def _ref(system: str) -> StoWfn:
    return StoWfn(str(EXAMPLES_DIR / system / 'stowfn.data'))


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('system', SYSTEMS)
def test_meta(system):
    """Header metadata must match exactly."""
    gen, ref = _convert(system), _ref(system)
    assert gen.periodicity == ref.periodicity
    assert gen.spin_unrestricted == ref.spin_unrestricted
    assert (gen.num_molorbs == ref.num_molorbs).all()
    assert gen.num_elec == ref.num_elec
    assert abs(gen.nuclear_repulsion_energy - ref.nuclear_repulsion_energy) < 1e-12


@pytest.mark.parametrize('system', SYSTEMS)
def test_geometry(system):
    """Atomic positions, numbers and valence charges must match."""
    gen, ref = _convert(system), _ref(system)
    assert gen.num_atom == ref.num_atom
    np.testing.assert_array_equal(gen.atomnum, ref.atomnum)
    np.testing.assert_allclose(gen.atompos, ref.atompos, atol=1e-12)
    np.testing.assert_allclose(gen.atomcharge, ref.atomcharge, atol=1e-12)


@pytest.mark.parametrize('system', SYSTEMS)
def test_basis(system):
    """Basis set (shell types, radial orders, exponents, AO counts) must match."""
    gen, ref = _convert(system), _ref(system)
    assert gen.num_shells == ref.num_shells
    assert gen.num_atorbs == ref.num_atorbs
    np.testing.assert_array_equal(gen.shelltype, ref.shelltype)
    np.testing.assert_array_equal(gen.order_r_in_shell, ref.order_r_in_shell)
    np.testing.assert_allclose(gen.zeta, ref.zeta, atol=1e-12)


@pytest.mark.parametrize('system', SYSTEMS)
def test_coefficients(system):
    """MO coefficients must agree within numerical precision.

    Known exceptions:
    - H: crashes in process_coefficients (open-shell matmul bug)
    """
    gen, ref = _convert(system), _ref(system)
    for sp in range(1 + int(gen.spin_unrestricted)):
        assert gen.coeff[sp] == pytest.approx(ref.coeff[sp])
