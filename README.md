adf2stowf
=========

[![PyPI version](https://img.shields.io/pypi/v/adf2stowf.svg)](https://pypi.org/project/adf2stowf/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adf2stowf.svg)](https://pypi.org/project/adf2stowf/)
[![Tests](https://github.com/Konjkov/adf2stowf/actions/workflows/tests.yml/badge.svg)](https://github.com/Konjkov/adf2stowf/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/adf2stowf/badge/?version=latest)](https://adf2stowf.readthedocs.io/en/latest/)

Converts wave function data from the **ADF** (Amsterdam Density Functional)
program into the `stowfn.data` input file for the **CASINO** quantum Monte
Carlo code.

ADF is the only major quantum chemistry program that uses Slater-Type Orbitals
(STO) natively. CASINO can use them directly by setting:

    atom_basis_type : slater-type

in the CASINO input file, which makes ADF+CASINO a powerful combination for
high-accuracy QMC calculations.

For general information about ADF, see https://www.scm.com/
For CASINO, see https://vallico.net/casinoqmc/


Requirements
============

* Python >= 3.10
* NumPy >= 2.0.0
* SciPy >= 1.13.1
* Matplotlib >= 3.9.0 (optional, for `--plot-cusps`)


Installation
============

```bash
pip install adf2stowf
```

or from source:

```bash
git clone https://github.com/Konjkov/adf2stowf
cd adf2stowf
pip install .
```


Usage
=====

1. Run ADF:

       adf < adf.in > adf.out

   This produces a binary file `TAPE21` in the working directory.

2. Convert to ASCII format using the `dmpkf` utility (included with ADF):

       dmpkf TAPE21 > TAPE21.asc

3. Run the converter in the same directory:

       adf2stowf

   This reads `TAPE21.asc` and writes `stowfn.data`.


Command-line options
====================

| Option | Description |
|--------|-------------|
| `--cusp-method=enforce` | Apply cusp correction to active orbitals **(default)** |
| `--cusp-method=project` | Project out cusp-violating components |
| `--cusp-method=none` | Disable cusp correction |
| `--cart2harm-projection` | Enforce pure spherical harmonics via orthogonal projection |
| `--all-orbitals` | Include virtual orbitals (default: occupied only) |
| `--plot-cusps` | Plot cusp constraints (requires Matplotlib) |
| `--dump` | Write a text dump of TAPE21 to `TAPE21.txt` |


Cartesian-to-spherical warning
===============================

You may see a warning during conversion:

    WARNING: cartesian to spherical conversion for spin 0, orb 0 violated by 0.00063567

ADF computes MOs in a Cartesian basis (6 d-functions, 10 f-functions).
CASINO requires pure spherical harmonics (5 d, 7 f). The extra Cartesian
components (e.g. the s-type contamination x²+y²+z² in d-shells) are
unphysical in a spherical harmonic representation.

Use `--cart2harm-projection` to remove these components via an orthogonal
projection onto the pure spherical harmonic subspace. Without it, the
contaminating components are silently dropped, which may slightly affect
the total energy.


Accuracy
========

HF total energies (Hartree) for HF/QZ4P/Slater calculations. **ADF** is the
reference energy from the source file; **CASINO** is the variational Monte Carlo
energy from the converted `stowfn.data`. The last column shows the deviation in
units of the CASINO statistical uncertainty (σ).

The ADF energies in the table below were obtained with this setting.

| System | ADF (HF energy) | CASINO (VMC energy) | Δ/σ |
|--------|----------------:|--------------------:|-----|
| H      |    −0.49999985  |    −0.49999968 ± 0.00000009 | 1.9 |
| H₂     |    −1.13359570  |    −1.13353329 ± 0.00002835 | 2.2 |
| He     |    −2.86166638  |    −2.86169958 ± 0.00004902 | 0.7 |
| Be     |   −14.57289824  |   −14.57210311 ± 0.00018682 | 4.3 |
| N      |   −54.40446246  |   −54.40440785 ± 0.00044673 | 0.1 |
| HCN    |   -92.91410391  |   -92.90146785 ± 0.00205019 | 6.2 |
| Ne     |  −128.54688836  |  −128.54704358 ± 0.00071620 | 0.2 |
| O₃     |  −224.36156862  |  −224.35580398 ± 0.00177555 | 3.2 |
| Ar     |  −526.81670427  |  −526.81634824 ± 0.00198243 | 0.2 |
| Ga     | -1850.36983279  |  -1923.0(1)                 |     |
| Kr     | −2752.05365745  | −2752.06972157 ± 0.01671619 | 1.0 |
| Xe     | −7232.13699292  | -7232.08669813 ± 0.03351053 | 1.5 |

**Note on numerical integration accuracy.**
ADF evaluates integrals on a numerical atom-centered grid. At the default
`NUMERICALQUALITY good` setting the quadrature error can reach 1–2 mHa, making
the ADF total energy an unreliable reference for sub-mHa comparisons with CASINO.
Always use `NUMERICALQUALITY excellent` in the ADF input when benchmarking
against VMC energies.

A VMC calculation with a single Slater determinant should reproduce the HF
energy exactly. The large deviations for Be and O₃ indicate a conversion error
that needs to be investigated and fixed.


Testing
=======

Reference output files for all example systems are included in `examples/`.
To run the regression tests:

```bash
pip install pytest
pytest tests/test_regression.py -v
```

Each test runs the full conversion pipeline on `examples/<system>/TAPE21.asc`
and compares the result against the reference `stowfn.data`.


Documentation
=============

Full documentation including mathematical background (cusp conditions,
Cartesian-to-spherical transformation matrices) is available at
https://adf2stowf.readthedocs.io/en/latest/

To build locally:

```bash
pip install sphinx sphinx-rtd-theme
cd docs && make html
```
