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

For an example of using ADF as a source of orbitals for all-electron QMC, see
Nemec, Towler & Needs, *Benchmark all-electron ab initio quantum Monte Carlo
calculations for small molecules* ([arXiv:0908.2041](https://arxiv.org/pdf/0908.2041)).


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
| `--cusp-method=project` | Project out cusp-violating components **(default)** |
| `--cusp-method=enforce` | Apply cusp correction to active orbitals |
| `--cusp-method=none` | Disable cusp correction |
| `--all-orbitals` | Include virtual orbitals (default: occupied only) |
| `--plot-cusps` | Plot cusp constraints (requires Matplotlib) |
| `--dump` | Write a text dump of TAPE21 to `TAPE21.txt` |

By default (`project`) the converter removes the cusp-violating components of
each orbital so the wavefunction satisfies the nuclear cusp condition. In a
molecule the per-nucleus cusp also picks up a smooth background from the tails
of basis functions on neighbouring atoms, so the residual deviation can stay
large without affecting the variational energy — a single-determinant VMC run
still reproduces the HF energy.


Cartesian-to-spherical conversion
=================================

ADF computes MOs in a Cartesian basis (6 d-functions, 10 f-functions).
CASINO requires pure spherical harmonics (5 d, 7 f). The extra Cartesian
components are not unphysical: they are themselves Slater orbitals with the
radial prefactor raised by r². Specifically:

- the s-type component x²+y²+z² of a d-shell is an s-type STO with radial prefactor r<sup>n+2</sup>
- the p-type components x·r², y·r², z·r² of an f-shell are p-type STOs with radial prefactor r<sup>n+2</sup>

The converter therefore represents each d/f shell **exactly** by appending a
companion shell with radial prefactor r<sup>n+2</sup> and the same zeta — no
Cartesian component is lost.

A subtlety of this transformation is normalisation: ADF MO coefficients refer
to individually normalised Cartesian monomials (the `bnorm` factors stored in
TAPE21), while CASINO expects coefficients of its own normalised real
harmonics. Within a d or f shell these norms differ between components, so the
polynomial transformation is conjugated by them,
`diag(1/casino_norm) · cart2harm · diag(bnorm)`. Omitting this conjugation
distorts molecular orbitals that mix d/f with s/p functions — invisible for
isolated atoms (closed and half-filled subshells are unitary-invariant) but
worth several mHa in molecules such as HCN or O₃.

Unused basis functions are pruned. Any shell — of any angular momentum
(s, p, d, or f), including the appended companion shells — whose coefficients
are zero in every written orbital is omitted from `stowfn.data`. These are
typically the polarisation d/f functions and diffuse s/p functions that no
occupied orbital uses; dropping them leaves the wavefunction unchanged while
reducing the number of basis functions CASINO must evaluate (for atoms this
can roughly halve the AO count). Pass `--all-orbitals` to keep them — the
virtual orbitals make use of them.


Accuracy
========

HF total energies (Hartree) for HF/QZ4P/Slater calculations. **ADF** is the
reference energy from the source file; **CASINO** is the variational Monte Carlo
energy from the converted `stowfn.data`. The **Reference HF** column gives
numerical Roothaan–Hartree–Fock energies from Bunge, Barrientos & Bunge,
*Atomic Data and Nuclear Data Tables* **53**, 113 (1993)
([doi:10.1006/adnd.1993.1003](https://doi.org/10.1006/adnd.1993.1003)),
accurate to 8–10 significant figures, for ground-state atoms He–Xe expressed
in a Slater-type orbital basis. The Δ/σ column shows the deviation between
ADF and CASINO in units of the CASINO statistical uncertainty (σ).

The ADF energies in the table below were obtained with the accuracy settings
described in the note below the table.

| System | Reference HF | ADF (HF energy) | ADF (basis) | CASINO (VMC energy) | Δ/σ |
|--------|-------------:|----------------:|:-----------:|--------------------:|-----|
| H      |              |    -0.49999985  | QZ4P |    -0.49999978 ± 0.00000010 | 0.7 |
| H₂     |              |    −1.13359570  | QZ4P |    -1.13357627 ± 0.00002838 | 0.7 |
| He     | -2.861679993 |    -2.86166638  | QZ4P |    -2.86167262 ± 0.00004938 | 0.1 |
| Be     | -14.57302313 |   -14.57301106  | QZ4P |   -14.57312996 ± 0.00018460 | 0.6 |
| B      | -24.52906069 |   -24.53271345  | pVQZ |   -24.53283773 ± 0.00025539 | 0.5 |
| C      | -37.68861890 |   -37.69324989  | pVQZ |   -37.69325944 ± 0.00031975 | 0.0 |
| N      | -54.40093415 |   -54.40446246  | QZ4P |   -54.40451476 ± 0.00045645 | 0.1 |
| CN⁻    |              |   -92.34646280  | pVQZ |   -92.34652077 ± 0.00063852 | 0.1 |
| HCN    |              |   -92.91263786  | mix  |   -92.91268420 ± 0.00062088 | 0.1 |
| Ne     | -128.5470980 |  −128.54688836  | QZ4P |  -128.54608391 ± 0.00071618 | 1.1 |
| O₃     |              |  −224.36156862  | QZ4P |  -224.36217991 ± 0.00098628 | 0.6 |
| Ar     | -526.8175122 |  −526.81670427  | QZ4P |  -526.81743472 ± 0.00199899 | 0.5 |
| Ga     | -1923.261001 | -1923.26303777  | QZ4P | -1923.26059272 ± 0.01267982 | 0.2 | x10
| Kr     | -2752.054969 | −2752.05365745  | QZ4P | -2752.05285347 ± 0.00538570 | 0.1 |
| Xe     | -7232.138349 | −7232.13699292  | QZ4P | -7232.12063576 ± 0.03401843 | 0.5 | x10

**Note on ADF accuracy settings.**
For a sub-mHa comparison with VMC the ADF input must contain

```
NUMERICALQUALITY excellent
```

(the default grid quality leaves 1–2 mHa of quadrature error), and in some
cases — all-electron calculations with a tight, near-linearly-dependent core
basis (e.g. Be in QZ4P), where the default pair-fit exchange leaves the SCF
~1 mHa above the true basis-set minimum — also

```
RIHartreeFock
  UseMe True
  Quality Excellent
  DependencyThreshold 1.0E-8
End
```

Note that the `RIHartreeFock` block is inert without `UseMe True`, and the
default `DependencyThreshold 1e-3` must be lowered — it removes exactly the
tight core combinations at issue.

A VMC calculation with a single Slater determinant should reproduce the HF energy
exactly; all systems in the table agree within statistics.


Verification
============

Correctness is verified by comparing the CASINO VMC energy of the converted
`stowfn.data` against the ADF reference energy: a single-determinant VMC run
must reproduce the HF energy. Reference inputs/outputs for all example systems
are included in `examples/` (see the table above).


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
