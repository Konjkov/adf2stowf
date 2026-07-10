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

## Working agreement (division of labor)

**Claude edits the source only. The user deploys and verifies.**

- Claude's job: edit `adf2stowf/*.py` (the source tree), reason about the change, and
  run **in-process checks of individual functions** (import the module, call methods, and
  inspect intermediate quantities — matrices, weights, identities — as in scratch scripts).
- The user's job: deploy the source to the test environment, regenerate `stowfn.data`
  for the example systems, run CASINO, and report the energies.
- Do **not** generate/overwrite the project's `stowfn.data` fixtures, reinstall the
  package, or launch CASINO. CASINO VMC energy vs the ADF HF reference is the ground
  truth; Claude's in-process numerical models are aids only and have been wrong before.
- Verification protocol: test the cheapest discriminating system first (e.g. an atom with
  occupied d such as **Ga**); if its energy is wrong, stop — do not test the rest.
- Note: the venv may hold a stale non-editable copy at `site-packages/adf2stowf/` that
  shadows the source. Deployment is the user's responsibility.

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
- `project` (default): project out cusp-violating components via null-space projection
- `enforce`: linear transformation of orbital coefficients to force cusp
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

The extra Cartesian components are themselves Slater orbitals with the radial
prefactor raised by r²: the `x²+y²+z²` component of a d-shell is an s orbital
(`r²·r^order_r`), and the `x·r²`, `y·r²`, `z·r²` components of an f-shell are p
orbitals. The conversion is therefore **exact** and is the only method: for every
Cartesian d/f valence shell the converter appends a companion shell with
`order_r + 2` and the same zeta (d → +s, f → +p), built in
`_build_extended_valence_basis`. The full square `cart2harm_map[st]` block routes
the harmonic rows to the parent shell and the contamination rows to the companion
shell, so `harm2cart ∘ cart2harm = I` and no Cartesian component is lost.

Normalisation (important): ADF stores MO coefficients relative to **individually
normalised Cartesian monomials** — `bnorm` is **per basis function, not per shell**
(within a d/f shell the components differ, e.g. `xy` vs `xx` by √3). CASINO instead
wants coefficients of its own `get_norm`-normalised real harmonics. So the bare
polynomial map must be conjugated by both norms: the d/f block in
`process_coefficients` is `diag(1/get_norm) · cart2harm_map[st] · diag(bnorm)`
(the radial factor cancels between parent and contamination rows, since they share
`n = l + order_r + 1`, so the conjugation is purely angular). For s/p this reduces
to the identity (`bnorm == get_norm`, one component per shell), which is why
omitting it was invisible for s/p and for spherical atoms (closed/half-filled
subshells are unitary-invariant) but distorted molecular d/f-bearing MOs by several
mHa (HCN, CN⁻, O₃). Do **not** transpose `cart2harm_map[st]` in this block — a
spurious `.T` passes a self-cancelling "identity check" yet destroys the orbitals.

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

5. **Cartesian → spherical is exact.** The rows of `cart2harm_map[st]` beyond
   `n_harm` (the contamination components) are routed to an appended companion
   shell (`order_r + 2`, same zeta) instead of being dropped, so the full square
   block is used and the conversion loses nothing. See `_build_extended_valence_basis`.

6. **Cusp enforcing** (`--cusp-method=enforce`) uses a linear operator derived in
   `stowfn.py:cusp_enforcing_matrix()` that modifies s-type orbital coefficients
   at each nucleus to satisfy the cusp condition exactly. Any changes to cusp math
   must be validated against all examples/.

   **Never snap coefficients to zero by magnitude after the cusp correction.** The
   cusp constraint matrix has entries up to ~1e6 (the tightest core-s function of a
   heavy atom, `norm·(Z−ζ)`), so a genuine coefficient as small as ~1e-11 can carry
   a ~1e-4 cusp contribution. A `|coeff| < 1e-10` snap zeroed exactly such a core-s
   coefficient in Ga's HOMO and re-broke the cusp (CASINO `STOWFDET_CUSP_CHECK`
   warnings up to 1.4e-4). To clean the ~1e-16 noise the dense cross-centre
   projection leaves in structurally-zero entries, record the `coeff == 0` mask
   **before** the projection and restore only those entries afterwards
   (`apply_cusp_correction`). CASINO checks the same absolute constraint
   `|Σ cusp_constraint_matrix·coeff| < 1e-8` (`stowfdet.f90`), not a relative one.

7. **Units**: all positions in ADF TAPE21 are in bohr (atomic units). CASINO also
   uses bohr. No unit conversion is needed.

8. **Dummy atoms** (ghost atoms in ADF) are excluded from the centre list —
   only real atoms (`[:Natoms]`) are written to stowfn.data.

9. **Empty-shell pruning** (`prune_empty_shells`, runs in `run()` after
   `finalize_coefficients`): drops any shell — of any l, including appended
   companion shells — whose coefficients are zero (`|c| < 1e-10`) in every written
   MO. These are unused polarisation d/f and diffuse s/p functions; pruning leaves
   the wavefunction unchanged but reduces the AO count CASINO evaluates (for atoms
   roughly halves it). `--all-orbitals` keeps them, since virtuals use them.

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

## ADF Accuracy Settings (required for sub-mHa VMC comparison)

The ADF total energy is a valid sub-mHa reference for CASINO only if the ADF input
contains:

```
NUMERICALQUALITY excellent
```

(the default grid quality leaves 1–2 mHa of quadrature error), and for all-electron
runs with a tight, near-linearly-dependent core basis (e.g. **Be/QZ4P**) also:

```
RIHartreeFock
  UseMe True
  Quality Excellent
  DependencyThreshold 1.0E-8
End
```

Found 2026-07 on Be/QZ4P (was ~1 mHa CASINO-above-ADF at 4σ; now 0.6σ, ADF
−14.57301106 = numerical HF limit −14.57302317 + 12 µHa):

- ADF's default **pair-fit** HF exchange cannot represent the exchange on the
  ill-conditioned QZ4P core: the SCF converges to orbitals ~1 mHa above the true
  basis-set minimum. CASINO's exact ⟨H⟩ of those orbitals was honest all along —
  such a gap is an ADF input problem, **not a converter bug**.
- The `RIHartreeFock` block is **inert without `UseMe True`** (default False); its
  other subkeys are silently ignored.
- `UseMe True` alone makes it *worse*: the scheme's default `DependencyThreshold
  1e-3` removes near-linearly-dependent combinations from the exchange matrix —
  exactly the tight core-s subspace at issue. Lower it to 1e-8.
- **`DependencyThreshold 1e-8` is ATOM-ONLY.** In molecules (seen on HCN/QZ4P,
  even at 1e-5) the cross-centre overlap of diffuse functions creates genuine
  near-linear dependence that must stay removed: the SCF oscillates wildly
  (errors up to ~660 a.u.) and "converges" to garbage (−2088.65 vs expected
  ≈ −92.91). This is the failure mode the ADF manual documents for this key
  ("unphysically large bond energy → raise DependencyThreshold"). For molecules
  keep the default 1e-3 (omit the line).
- **QZ4P itself is atom-oriented.** In molecules it does not give an accurate
  wavefunction: either the automatic dependency truncation (`Dependency
  bas=4e-3`, auto-on for HF) distorts the basis, or keeping it intact
  destabilizes the SCF. Recipe split: atoms → QZ4P + RIHartreeFock block above;
  molecules → a well-conditioned pVQZ-based basis (the "mix" basis for HCN in
  the README table, ADF↔CASINO agreement 0.1σ).

---

## Common Pitfalls

- If `froc_A` values sum to a non-integer, ADF used fractional occupation (smearing).
  The code handles this but may print "leftover partial occupation" warnings.
- `HOMO > LUMO` warnings are expected for some open-shell or smeared calculations.
- D/F core orbitals (`shelltype 4/5`) raise `NotImplementedError` — not yet supported.
- Running without `dmpkf` produces a binary TAPE21, not TAPE21.asc — the parser only
  reads the ASCII dump.
