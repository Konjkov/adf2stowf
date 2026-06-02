---
name: quantum-chem-adf2stowf
description: >
  Use this skill whenever working with the adf2stowf project or any task involving
  ADF (Amsterdam Density Functional) output files, CASINO quantum Monte Carlo input files,
  Slater-type orbitals (STO), cusp conditions, or conversion between Cartesian and
  spherical harmonic basis functions. Trigger on any mention of: TAPE21, stowfn.data,
  STO basis sets, ADF, CASINO, cusp correction, cart2harm, nuclear cusp, zeta exponents,
  or quantum chemistry wavefunction conversion.
---

# Quantum Chemistry: adf2stowf Project

## Project Overview

This project converts molecular orbital wavefunctions from **ADF** (Amsterdam Density
Functional, SCM) into **CASINO** (quantum Monte Carlo) input format. ADF is the only
major QC program that uses Slater-type orbitals (STO) natively; CASINO can use them
directly with `atom_basis_type: slater-type`.

**Pipeline:**
```
adf < adf.in > adf.out   →   TAPE21 (binary KF file)
dmpkf TAPE21 > TAPE21.asc  →   TAPE21.asc (ASCII KF dump)
adf2stowf                  →   stowfn.data (CASINO input)
```

**Source:** https://www.scm.com/doc/ADF/kf_defs.html

---

## Physics Background

### Slater-Type Orbitals (STO)

STO basis functions have the form:

```
χ(r) = N · r^(n-l-1) · exp(-ζ·r) · Y_l^m(θ,φ)
```

where:
- `n` = principal quantum number (`nqbas` / `nqcor` in TAPE21)
- `l` = angular momentum quantum number (`lqbas` / `lqcor` in TAPE21)
- `ζ` (zeta) = exponential decay parameter (`alfbas` / `alfcor` in TAPE21)
- `r^(n-l-1)` = radial prefactor, order stored as `order_r = n - l - 1`
- `Y_l^m` = real spherical harmonic

**Key difference from GTO:** STOs decay as `exp(-ζr)` (correct physical behaviour),
Gaussians decay as `exp(-αr²)`. STOs satisfy the nuclear cusp condition exactly for
hydrogen-like atoms; GTOs do not.

### Nuclear Cusp Condition

At a nucleus with charge Z, the exact wavefunction must satisfy:

```
(dψ/dr)|_{r=0} = -Z · ψ(0)
```

ADF orbitals computed in a Cartesian basis may violate this. The converter applies
one of three corrections (CLI flag `--cusp-method`):
- `enforce` (default): linear transformation of orbital coefficients to force cusp
- `project`: project out cusp-violating components via null-space projection
- `none`: no correction (use only if cusps are already satisfied)

### Shell Types (ADF → CASINO encoding)

ADF stores `lqbas` (angular momentum l = 0,1,2,3 for s,p,d,f). The code maps this
to an internal `shelltype` integer used by CASINO's stowfn.data format:

| l | Shell | shelltype | Nharmonics | Ncartesian |
|---|-------|-----------|------------|------------|
| 0 | s     | 1         | 1          | 1          |
| 1 | p     | 3         | 3          | 3          |
| 2 | d     | 4         | 5          | 6          |
| 3 | f     | 5         | 7          | 10         |

Note: `shelltype = l + 1 + (l > 0)` — the gap between p(3) and d(4) encodes that
sp-shells (shelltype=2) are a different concept not used here.

### Cartesian vs Spherical Harmonics

ADF internally uses **Cartesian** d and f functions (6 d-functions, 10 f-functions),
but STOs for CASINO must be expressed in **pure spherical harmonics** (5 d, 7 f).

The extra Cartesian components (e.g., `x²+y²+z²` for d-shells, which has s-symmetry)
are non-physical in a spherical harmonic context. The transformation matrices
`harm2cart_map` and `cart2harm_map` handle this. If Cartesian orbitals contain
s-type contamination in d-shells, a warning is printed:

```
WARNING: cartesian to spherical conversion for spin 0, orb 0 violated by 0.00063567
```

Use `--cart2harm-projection` to remove this contamination via SVD-based null-space
projection (numerically stable, preferred over pseudoinverse).

---

## Key Data Structures (TAPE21.asc → Python)

### Geometry section
| TAPE21 key | Python attribute | Meaning |
|-----------|-----------------|---------|
| `nnuc` | `Natoms` | Number of real atoms |
| `ntyp` | `Natomtypes` | Number of distinct atom types |
| `xyz` | `Geometry['xyz']` | Atomic positions in bohr (au) |
| `atomtype total charge` | `atomicnumber_per_atomtype` | Nuclear charge Z per atom type |
| `fragment and atomtype index` | `atyp_idx` | Maps each atom → atom type index |

### Basis section (valence STO)
| TAPE21 key | Python attribute | Meaning |
|-----------|-----------------|---------|
| `nbset` | `nbset` | Total number of valence STO shells |
| `nqbas` | `nqbas` | Principal quantum number n per shell |
| `lqbas` | `lqbas` | Angular momentum l per shell |
| `alfbas` | `valence_zeta` | Exponent ζ per shell |
| `bnorm` | `valence_cartnorm` | Cartesian normalisation per basis fn |
| `nbos` | `nbos` | Total number of Cartesian basis fns |
| `nbaspt` | `nbaspt` | Shell pointer: first shell of each atomtype |
| `nbptr` | `nbptr` | Cartesian fn pointer: first fn of each atomtype |

Derived: `valence_order_r = nqbas - lqbas - 1` (exponent of radial prefactor `r`)

### Core section (frozen core STO)
| TAPE21 key | Python attribute | Meaning |
|-----------|-----------------|---------|
| `ncset` | `ncset` | Total number of core STO shells |
| `nqcor` | `nqcor` | Principal quantum number n (core) |
| `lqcor` | `lqcor` | Angular momentum l (core) |
| `alfcor` | `core_zeta` | Exponent ζ (core) |
| `cornrm` | `core_cartnorm` | Cartesian normalisation (core) |
| `nrcset` | `nrcset` | Shape (Natomtypes, 4): count of core shells per l |
| `nrcorb` | `nrcorb` | Shape (Natomtypes, 4): count of core MOs per l |
| `ccor` | `ccor` | Core MO coefficients (in STO basis) |

### Symmetry / MO coefficients
| TAPE21 key | Meaning |
|-----------|---------|
| `symlab` | List of symmetry labels (e.g. `['SIGMA', 'PI']` or `['A1', 'E1']`) |
| `nsym` | Number of symmetry irreps |
| `norb` | Number of MOs per irrep |
| `Eigen-Bas_A` / `_B` | MO coefficients (spin A/B) in Cartesian STO basis |
| `eps_A` / `_B` | MO eigenvalues (orbital energies) in Hartree |
| `froc_A` / `_B` | Fractional occupation numbers |
| `npart` | Basis function indices (1-based) active in this symmetry |

Spin A = alpha (or restricted), Spin B = beta (unrestricted only).
`Nspins = 1` → spin restricted; `Nspins = 2` → spin unrestricted.

---

## stowfn.data Format (CASINO)

The output file is read by CASINO with `atom_basis_type: slater-type`.
Key sections written by `StoWfn.writefile()`:

- `BASIC INFO`: periodicity, spin_unrestricted, nuclear repulsion energy, Nelectrons
- `GEOMETRY`: atomic positions (bohr), atomic numbers, valence charges
- `BASIS SET`: centre positions, shell types, `order_r`, zeta exponents
- `MO COEFFICIENTS`: expansion of each MO in the STO basis

Shell type encoding in stowfn.data: `s=1, sp=2, p=3, d=4, f=5, g=6`
(same as `shelltype` in the Python code)

---

## Important Implementation Notes

1. **ADF uses 1-based indexing** in TAPE21; all pointer arrays (`nbaspt`, `nbptr`,
   `ncorpt`, `npart`) are decremented by 1 on read.

2. **Core orbitals are atom-centred** and stored per atom type (not per atom).
   The converter reconstructs per-atom core MOs by tiling over `atyp_idx`.

3. **D/F shell `harm2cart_map` matrices** are hardcoded from CASINO's `stowfdet.f90`
   polynomial ordering — do NOT change without cross-checking CASINO source.

4. **Partial occupations** (fractional occupation in ADF, e.g. from smearing) are
   accumulated by eigenvalue and redistributed; a leftover warning is printed if any
   remain after all orbitals are processed.

5. **`cart2harm_constraint`** captures the rows of `cart2harm_map` corresponding to
   non-physical Cartesian components (rows beyond `n_harm`). These must be zero for
   pure spherical harmonics. Violation norm > 1e-5 triggers a warning.

6. **Cusp enforcing** (`--cusp-method=enforce`) uses a linear operator derived in
   `stowfn.py:cusp_enforcing_matrix()` that modifies s-type orbital coefficients
   at each nucleus to satisfy the cusp condition exactly. Any changes to cusp math
   must be validated against all examples/.

7. **Units**: all positions in ADF TAPE21 are in bohr (atomic units). CASINO also
   uses bohr. No unit conversion is needed.

8. **Dummy atoms** (ghost atoms in ADF) are excluded from the centre list —
   only real atoms (`[:Natoms]`) are written to stowfn.data.

---

## File Layout

```
adf2stowf/
├── adf2stowf/
│   ├── adf2stowf.py   # Main converter: ADFToStoWF class, CLI entry point
│   ├── adfread.py     # AdfParser: reads TAPE21.asc into nested dict
│   └── stowfn.py      # StoWfn: reads/writes stowfn.data, cusp math
├── kfreader/
│   └── kf.py          # Alternative binary KF reader (not used in main flow)
└── examples/          # Reference inputs/outputs: H, He, Be, N, Ne, Ar, Kr, Xe, O3, H2
```

---

## Common Pitfalls

- If `froc_A` values sum to a non-integer, ADF used fractional occupation (smearing).
  The code handles this but may print "leftover partial occupation" warnings.
- `HOMO > LUMO` warnings are expected for some open-shell or smeared calculations.
- D/F core orbitals (`shelltype 4/5`) raise `NotImplementedError` — not yet supported.
- Running without `dmpkf` produces a binary TAPE21, not TAPE21.asc — the parser only
  reads the ASCII dump.
