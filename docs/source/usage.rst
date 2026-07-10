Usage
=====

Workflow
--------

**Step 1.** Run the ADF program::

    adf < adf.in > adf.out

This leaves a binary file ``TAPE21`` in the working directory.

**Step 2.** Convert the binary file to ASCII format::

    dmpkf TAPE21 > TAPE21.asc

The ``dmpkf`` utility is included with the ADF distribution.

**Step 3.** Run the converter in the same directory::

    adf2stowf

This reads ``TAPE21.asc`` and writes ``stowfn.data``.

**Step 4.** Use the output in CASINO by setting::

    atom_basis_type   : slater-type

in the CASINO input file.

ADF Accuracy Settings
---------------------

For a sub-mHa comparison between the ADF HF energy and CASINO VMC the ADF
input must contain::

    NUMERICALQUALITY excellent

(the default grid quality leaves 1–2 mHa of quadrature error), and in some
cases — all-electron calculations with a tight, near-linearly-dependent core
basis (e.g. Be in QZ4P), where ADF's default pair-fit exchange leaves the SCF
about 1 mHa above the true basis-set minimum — also::

    RIHartreeFock
      UseMe True
      Quality Excellent
      DependencyThreshold 1.0E-8
    End

Note that the ``RIHartreeFock`` block is inert without ``UseMe True``, and the
default ``DependencyThreshold`` of 1e-3 must be lowered — it removes exactly
the tight core combinations at issue.

``DependencyThreshold 1e-8`` is for **atoms only**.  In molecules the
cross-centre overlap of diffuse QZ4P functions creates genuine near-linear
dependence that must stay removed: with a tiny threshold the SCF becomes
unstable and converges to an unphysical energy.  For molecules keep the
default (omit the line).

More generally, QZ4P itself is atom-oriented and does not give an accurate
wavefunction in molecules — either the automatic dependency truncation
distorts the basis or keeping it intact destabilizes the SCF.  For molecules
use a well-conditioned pVQZ-based basis instead.

Command-Line Options
--------------------

.. option:: adf2stowf

    Run with defaults (equivalent to ``--cusp-method=project``).

.. option:: --cusp-method=project

    Project out cusp-violating components from each orbital. **Default.**
    See :ref:`cusp-conditions` in the :doc:`adf2stowf` module documentation.

.. option:: --cusp-method=enforce

    Apply cusp correction to active orbitals.
    See :ref:`cusp-conditions` in the :doc:`adf2stowf` module documentation.

.. option:: --cusp-method=none

    Disable cusp correction entirely.
    See :ref:`cusp-conditions` in the :doc:`adf2stowf` module documentation.

.. option:: --plot-cusps

    Enable plotting of cusp constraints (requires Matplotlib).

.. option:: --all-orbitals

    Include virtual orbitals in addition to occupied ones.
    Default: occupied orbitals only.

.. option:: --dump

    Generate a text dump of ``TAPE21.asc``.

Notes
-----

The Cartesian-to-spherical conversion is exact — no components are discarded
(see :ref:`cart-to-harm`).  In a molecule the per-nucleus cusp condition also
picks up a smooth background from the tails of basis functions centred on
neighbouring atoms, so the residual cusp deviation reported during conversion
can stay large without affecting the variational energy: a single-determinant
VMC run still reproduces the HF energy.

Unused basis functions are pruned.  Any shell — of any angular momentum
(s, p, d, or f), including the appended companion shells — whose coefficients
are zero in every written orbital is omitted from ``stowfn.data``.  These are
typically the polarisation d/f functions and diffuse s/p functions that no
occupied orbital uses; dropping them leaves the wavefunction unchanged while
reducing the number of basis functions CASINO must evaluate.  Use
``--all-orbitals`` to keep them, since the virtual orbitals make use of them.
