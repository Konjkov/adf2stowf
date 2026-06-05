"""VMC energy test: compare CASINO output energies against ADF HF reference.

For each system that has a casino/out file, parses the VMC energy (On-the-fly
reblocking method) and the ADF total HF energy from TAPE21.asc, then checks
that the deviation does not exceed max(3σ, atol), where atol accounts for the
correlation energy difference between HF and VMC.

The loose absolute tolerance (0.01 au per electron) is intentional: the VMC
wave function is a pure Slater determinant (no Jastrow), so it recovers the
HF energy but not the correlation energy. The test is designed to catch
gross errors in the conversion (wrong normalization, missing orbitals, etc.)
rather than sub-mHartree accuracy.

Run with::

    pytest tests/test_vmc_energy.py -v
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / 'examples'
sys.path.insert(0, str(REPO_ROOT))

from adf2stowf.adfread import AdfParser  # noqa: E402

# Systems that have both TAPE21.asc and a casino/out file
SYSTEMS = sorted(
    d.name
    for d in EXAMPLES_DIR.iterdir()
    if d.is_dir()
    and (d / 'TAPE21.asc').exists()
    and (d / 'casino' / 'out').exists()
)

_VMC_LINE = re.compile(
    r'^\s*([+-]?\d+\.\d+)\s+\+/-\s+(\d+\.\d+)\s+On-the-fly reblocking method'
)


def _parse_casino_energy(system: str) -> tuple[float, float]:
    """Return (vmc_energy, stderr) from casino/out (On-the-fly reblocking)."""
    out = (EXAMPLES_DIR / system / 'casino' / 'out').read_text()
    for line in out.splitlines():
        m = _VMC_LINE.match(line)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise ValueError(f'No On-the-fly reblocking line found in {system}/casino/out')


def _parse_adf_energy(system: str) -> tuple[float, int]:
    """Return (total HF energy, num_electrons) from TAPE21.asc."""
    import os
    orig = os.getcwd()
    try:
        os.chdir(EXAMPLES_DIR / system)
        data = AdfParser('TAPE21.asc').parse()
    finally:
        os.chdir(orig)
    energy = float(data['Total Energy']['Total energy'][0])
    num_elec = int(round(float(data['General']['electrons'][0])))
    return energy, num_elec


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('system', SYSTEMS)
def test_vmc_energy(system):
    """VMC energy must agree with ADF HF energy within max(3σ, 0.01*N_elec) au.

    The 0.01 au/electron tolerance accounts for the correlation energy not
    recovered by a bare Slater-determinant VMC wave function.  It is loose
    enough to pass physically correct results while catching gross conversion
    errors (factor-of-cornrm mistakes, wrong orbital counts, etc.).
    """
    vmc, stderr = _parse_casino_energy(system)
    adf, n_elec = _parse_adf_energy(system)
    deviation = abs(vmc - adf)
    tol = max(3 * stderr, 0.01 * n_elec)
    assert deviation < tol, (
        f'{system}: |VMC - ADF| = {deviation:.6f} au '
        f'exceeds tol = {tol:.6f} au '
        f'(VMC={vmc}, ADF={adf}, σ={stderr}, N_elec={n_elec})'
    )

