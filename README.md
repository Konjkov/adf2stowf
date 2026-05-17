adf2stowf
=========

[![PyPI version](https://img.shields.io/pypi/v/adf2stowf.svg)](https://pypi.org/project/adf2stowf/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adf2stowf.svg)](https://pypi.org/project/adf2stowf/)
[![Tests](https://github.com/Konjkov/adf2stowf/actions/workflows/tests.yml/badge.svg)](https://github.com/Konjkov/adf2stowf/actions/workflows/tests.yml)

Converts wave function data from the **ADF** (Amsterdam Density Functional)
program into the `stowfn.data` input file for the **CASINO** quantum Monte
Carlo code.

ADF is the only major quantum chemistry program that uses Slater-Type Orbitals
(STO) natively. CASINO can use them directly by setting:

    atom_basis_type : slater-type

in the CASINO input file, which makes ADF+CASINO a powerful combination for
high-accuracy QMC calculations.

For general information about ADF, see https://www.scm.com/
For CASINO, see https://casino.ph.utexas.edu/


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
Cartesian-to-spherical transformation matrices) is available in `docs/`.
To build locally:

```bash
pip install sphinx sphinx-rtd-theme
cd docs && make html
```
