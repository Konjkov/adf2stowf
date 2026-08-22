Changelog
=========

1.1.0
-----

Added

* Exact Cartesian-to-spherical conversion: the Cartesian components that are not
  spanned by the pure spherical harmonics are no longer discarded but written as
  companion shells (d -> extra s, f -> extra p), so no part of the ADF basis is
  lost.
* Regression tests (`tests/test_regression.py`) converting every
  `examples/*/TAPE21.asc` and comparing against the committed `stowfn.data`.
  The GitHub Actions job now actually runs them on Python 3.10-3.14.
* New examples: B, Be, C, CN-, Ga, HCN.
* `plot` optional extra for `--plot-cusps`: `pip install adf2stowf[plot]`.
* MIT license file and license metadata.

Fixed

* Cusp correction: only the structural zeros recorded before the projection are
  restored, instead of thresholding written coefficients by magnitude. The cusp
  constraint matrix has entries up to ~1e6, so a genuine ~1e-11 coefficient can
  carry a ~1e-4 cusp contribution; the old `|coeff| < 1e-10` snap zeroed exactly
  such a coefficient in Ga's HOMO and triggered CASINO `STOWFDET_CUSP_CHECK`
  warnings.
* Regression tests no longer depend on the CPU the runner happens to use: values
  written in fixed 20-character columns stick to a negative left neighbour, and
  whitespace tokenising compared such glued pairs as strings, so a last-bit
  difference from another BLAS kernel failed the comparison. The matrix no
  longer uses `fail-fast`, so one failing Python version does not cancel the
  others.
* Removed the misleading nuclear-cusp basis warning: in molecules the per-nucleus
  deviation stays large because of the smooth background from neighbouring atoms'
  basis tails while the variational energy is unaffected, so the warning was a
  false positive advising an unnecessary basis change.

Changed

* Matplotlib is no longer a hard dependency; it is imported only for
  `--plot-cusps`.
* The d/f output normalisation tables are derived from a single
  `stowfn.POLYNORM` constant instead of a hand-copied array.
* Documentation and README updated for the exact d/f conversion and the current
  dependency versions.

1.0.0
-----

First production release.
