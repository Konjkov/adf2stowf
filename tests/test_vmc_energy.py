"""VMC energy test: compare CASINO output energies against ADF HF reference.

For each system that has both a casino/out and an output.dat file, parses
the VMC energy (On-the-fly reblocking method) and the ADF total HF energy,
then checks that the deviation does not exceed 3σ.

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

# Systems that have both casino/out and output.dat
SYSTEMS = sorted(
    d.name
    for d in EXAMPLES_DIR.iterdir()
    if d.is_dir()
    and (d / 'casino' / 'out').exists()
    and (d / 'output.dat').exists()
)

_VMC_LINE = re.compile(
    r'^\s*([+-]?\d+\.\d+)\s+\+/-\s+(\d+\.\d+)\s+On-the-fly reblocking method'
)
_ADF_ENERGY_LINE = re.compile(
    r'Total energy\s+([+-]?\d+\.\d+)\s+a\.u\.'
)


def _parse_casino_energy(system: str) -> tuple[float, float]:
    """Return (vmc_energy, stderr) from casino/out (On-the-fly reblocking)."""
    out = (EXAMPLES_DIR / system / 'casino' / 'out').read_text()
    for line in out.splitlines():
        m = _VMC_LINE.match(line)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise ValueError(f'No On-the-fly reblocking line found in {system}/casino/out')


def _parse_adf_energy(system: str) -> float:
    """Return total HF energy from output.dat."""
    text = (EXAMPLES_DIR / system / 'output.dat').read_text()
    m = _ADF_ENERGY_LINE.search(text)
    if not m:
        raise ValueError(f'No "Total energy" line found in {system}/output.dat')
    return float(m.group(1))


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('system', SYSTEMS)
def test_vmc_energy(system):
    """VMC energy must agree with ADF HF energy within 3σ."""
    vmc, stderr = _parse_casino_energy(system)
    adf = _parse_adf_energy(system)
    deviation = abs(vmc - adf)
    assert deviation < 3 * stderr, (
        f'{system}: |VMC - ADF| = {deviation:.6f} au '
        f'exceeds 3σ = {3 * stderr:.6f} au '
        f'(VMC={vmc}, ADF={adf}, σ={stderr})'
    )

