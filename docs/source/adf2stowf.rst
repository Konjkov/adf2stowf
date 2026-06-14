adf2stowf — ADF to CASINO STO Wavefunction Converter
=====================================================

.. module:: adf2stowf
   :synopsis: Convert ADF TAPE21.asc output to CASINO stowfn.data format.

Overview
--------

This module converts a Slater-Type Orbital (STO) wavefunction from an ADF
(Amsterdam Density Functional) ``TAPE21.asc`` file into the ``stowfn.data``
format used by the CASINO quantum Monte Carlo code.

The conversion pipeline covers:

* Parsing geometry, basis set (valence + core), and MO coefficients from the
  ADF output via :class:`adfread.AdfParser`;
* Remapping ADF's Cartesian angular basis functions to the real solid-harmonic
  (spherical) polynomial basis expected by CASINO.  The extra Cartesian
  components of d/f shells are not discarded: each is itself a Slater orbital
  and is represented exactly by an appended companion shell (d → +s, f → +p),
  so the conversion is lossless;
* Enforcing nuclear cusp conditions on the resulting MO coefficients through
  one of three strategies: *enforce*, *project*, or *none*;
* Writing a ready-to-use :class:`stowfn.StoWfn` file and, optionally,
  plotting the wavefunction cusp behaviour per atom.

The module is intended to be run as a command-line script.  The
:class:`ADFToStoWF` class encapsulates all conversion logic and can also
be instantiated programmatically.


ADFToStoWF class
----------------

.. class:: ADFToStoWF(plot_cusps, cusp_method, do_dump, only_occupied)

   Top-level converter object.  Parses ``TAPE21.asc`` on construction and
   exposes the full conversion pipeline through :meth:`run`.

   :param plot_cusps: When ``True``, evaluate MO values and Laplacians along
      a z-axis line through each nucleus before and after cusp correction, and
      save a ``cusp_constraint.svg`` plot.
   :type plot_cusps: bool
   :param cusp_method: Strategy for enforcing nuclear cusp conditions.
      One of ``'enforce'``, ``'project'``, or ``'none'``
      (see :meth:`apply_cusp_correction`).
   :type cusp_method: str
   :param do_dump: When ``True``, write a human-readable ``TAPE21.txt`` dump
      of the parsed ADF data (via :meth:`adfread.AdfParser.write_dump`).
   :type do_dump: bool
   :param only_occupied: When ``True``, include only occupied MOs in the
      output (occupation >= 2/Nspins).  When ``False``, virtual orbitals
      are also written.
   :type only_occupied: bool

   .. rubric:: Key instance attributes

   **General / metadata** (from the ``General`` section of TAPE21)

   .. attribute:: Nspins
      :type: int

      Number of spin channels (1 = restricted, 2 = unrestricted).

   .. attribute:: spin_restricted
      :type: bool

      ``True`` when ``Nspins == 1``.

   .. attribute:: Nvalence_electrons
      :type: int

      Total number of valence electrons.

   .. attribute:: Natoms
      :type: int

      Number of real (non-dummy) atoms.

   .. attribute:: Natomtypes
      :type: int

      Number of distinct atom types.

   .. attribute:: atyp_idx
      :type: numpy.ndarray, shape (Natoms,)

      Zero-based atom-type index for each atom; maps atom positions to
      their corresponding atom type in the basis tables.

   **Valence basis** (populated by :meth:`process_valence_basis`)

   .. attribute:: nbset
      :type: int

      Total number of valence STO shells across all atom types.

   .. attribute:: valence_shelltype
      :type: numpy.ndarray

      Shell-type code for every valence shell (CASINO encoding:
      1 = s, 2 = sp, 3 = p, 4 = d, 5 = f).

   .. attribute:: valence_order_r
      :type: numpy.ndarray

      Radial power *N* for every valence shell (``nqbas - lqbas - 1``).

   .. attribute:: valence_zeta
      :type: numpy.ndarray

      Slater exponents for every valence shell.

   .. attribute:: valence_cartnorm
      :type: numpy.ndarray

      Cartesian normalization factors (``bnorm``) for every valence
      Cartesian basis function.

   **Core basis** (populated by :meth:`process_core_basis`)

   .. attribute:: ncset
      :type: int

      Total number of frozen-core STO shells across all atom types.

   .. attribute:: core_shelltype
      :type: numpy.ndarray

      Shell-type codes for every core shell.

   .. attribute:: core_order_r
      :type: numpy.ndarray

      Radial power *N* for every core shell.

   .. attribute:: core_zeta
      :type: numpy.ndarray

      Slater exponents for every core shell.

   **Transformation matrices** (built in :meth:`initialize_data`)

   .. attribute:: harm2cart_map
      :type: dict[int, numpy.ndarray]

      Mapping from shell-type code to the matrix **M** that converts
      *spherical* AO coefficients (rows) to *Cartesian* AO coefficients
      (columns).  Available for s (1), p (3), d (4), and f (5) shells;
      d and f matrices follow the CASINO ``stowfdet.f90`` polynomial
      ordering convention.

   .. attribute:: cart2harm_map
      :type: dict[int, numpy.ndarray]

      Inverse of :attr:`harm2cart_map` (computed via ``numpy.linalg.inv``).

   .. attribute:: cart2harm_matrix
      :type: numpy.ndarray, shape (Nharmbasfns, Nvalence_cartbasfn)

      Block-diagonal matrix that converts valence MO coefficients from the
      Cartesian AO basis (as stored in TAPE21) to the real solid-harmonic
      basis required by CASINO.  Each d/f block is conjugated by the per-function
      ADF and CASINO normalisations (see :ref:`cart-to-harm`).

   **MO coefficients** (populated by :meth:`process_coefficients` and :meth:`finalize_coefficients`)

   .. attribute:: molorb_cart_coeff
      :type: list of numpy.ndarray

      ``molorb_cart_coeff[spin]`` — valence MO coefficients in the Cartesian
      AO basis, shape ``(Nvalence_molorbs[spin], Nvalence_cartbasfn)``,
      sorted by eigenvalue.

   .. attribute:: valence_molorb_harm_coeff
      :type: list of numpy.ndarray

      Valence MO coefficients after the Cartesian-to-spherical transformation,
      shape ``(Nharmbasfns, Nvalence_molorbs[spin])`` per spin.

   .. attribute:: coeff
      :type: list of numpy.ndarray

      Final full MO coefficient matrices (core + valence combined, spherical
      basis), shape ``(Nharmbasfns, Nmolorbs[spin])`` per spin.

   .. attribute:: sto
      :type: stowfn.StoWfn

      The :class:`stowfn.StoWfn` object constructed in :meth:`setup_stowfn`
      and written to ``stowfn.data`` at the end of :meth:`apply_cusp_correction`.

   .. rubric:: Methods

   .. method:: initialize_data()

      Extract the primary data sections from the parsed TAPE21 dictionary and
      compute derived metadata.

      Populates ``General``, ``Geometry``, ``Basis``, ``Core``, and
      ``Symmetry`` section references, scalar counts (:attr:`Nspins`,
      :attr:`Natoms`, …), the atom-type index :attr:`atyp_idx`, and the
      :attr:`harm2cart_map` / :attr:`cart2harm_map` transformation matrices
      for s, p, d, and f shells.  Optionally writes ``TAPE21.txt`` when
      :attr:`DO_DUMP` is ``True``.

      :raises AssertionError: If atom or atom-type counts in the file are
         internally inconsistent.

   .. method:: process_valence_basis()

      Parse the ``Basis`` section of TAPE21 and populate all valence-basis
      attributes.

      Reads shell counts (``nbset``, ``nbaspt``), quantum numbers (``nqbas``,
      ``lqbas``), Slater exponents (``alfbas``), Cartesian function counts
      (``nbptr``), and normalization factors (``bnorm``).  Derives per-centre
      and per-atom-type slices for all of the above.

      :raises AssertionError: If any basis dimension is inconsistent with the
         geometry or the total number of AOs (``naos``).

   .. method:: process_core_basis()

      Parse the ``Core`` section of TAPE21 and populate all frozen-core-basis
      attributes.

      Reads shell counts (``ncset``, ``ncorpt``, ``nrcset``), quantum numbers
      (``nqcor``, ``lqcor``), Slater exponents (``alfcor``), and per-shell
      normalization factors (``cornrm``).  Expands normalization values into
      per-basis-function arrays for s and p core shells.

      :raises ValueError: If d- or f-type frozen core shells are encountered
         (not yet implemented).
      :raises AssertionError: If core-shell counts do not match the ``nrcset``
         table.

   .. method:: process_shells()

      Merge core and valence shell data into per-centre arrays.

      Concatenates core shells (from :meth:`process_core_basis`) before
      valence shells (from :meth:`process_valence_basis`) for each atom,
      producing ``shelltype_per_centre``, ``order_r_per_centre``, and
      ``zeta_per_centre``.

   .. method:: select_coeff(sp)

      Collect and sort MO coefficients for spin channel *sp* from the
      symmetry-resolved ``Eigen-Bas`` blocks in TAPE21.

      Iterates over all symmetry labels (``symlab``), reads fractional
      occupations (``froc_A`` / ``froc_B``), eigenvalues (``eps_A`` /
      ``eps_B``), and coefficient matrices (``Eigen-Bas_A`` /
      ``Eigen-Bas_B``).  Handles partial occupations by accumulating
      leftover occupation at degenerate eigenvalues.  Returns orbitals
      sorted by eigenvalue.

      Prints a ``Warning: HOMO > LUMO`` message if the highest occupied
      eigenvalue exceeds the lowest unoccupied eigenvalue, which can occur
      for open-shell or fractional-occupation calculations.

      :param sp: Spin channel index (0 = alpha, 1 = beta).
      :type sp: int
      :returns: MO coefficient matrix in the Cartesian AO basis.
      :rtype: numpy.ndarray, shape ``(Nmolorbs_selected, Nvalence_cartbasfn)``

   .. method:: process_coefficients()

      Build the Cartesian-to-spherical transformation and apply it to the MO
      coefficients.

      Assembles the block-diagonal :attr:`cart2harm_matrix` from the per-shell
      :attr:`cart2harm_map` entries.  For d/f shells the bare polynomial block
      is conjugated by the per-function normalisations,
      ``diag(1/get_norm) · cart2harm_map[st] · diag(bnorm)``, because ADF MO
      coefficients refer to individually normalised Cartesian monomials while
      CASINO expects normalised real harmonics (see :ref:`cart-to-harm`).  The
      full square block is used: the spherical-harmonic rows feed the parent
      shell and the contamination rows feed the appended companion shell, so no
      Cartesian component is lost.

      Applies :attr:`cart2harm_matrix` to produce
      :attr:`valence_molorb_harm_coeff`.

      :raises AssertionError: If basis-function counts are inconsistent
         after the transformation.

   .. method:: process_core_orbitals()

      Build the coefficient matrix for frozen-core MOs in the spherical basis.

      Reads the core-orbital expansion coefficients (``ccor``) and the number
      of core MOs per shell type (``nrcorb``).  Constructs
      ``core_molorb_coeff`` of shape ``(Nharmbasfns, Ncore_molorbs)`` by
      placing each core AO coefficient into the appropriate rows (atom's
      harmonic basis functions) and column (core MO index).

      Supports s- and p-type core shells only.

   .. method:: finalize_coefficients()

      Concatenate core and valence MO coefficient matrices into :attr:`coeff`
      and build the combined AO normalization array ``norm_per_harmbasfn``.

      After this step, ``coeff[spin]`` has shape
      ``(Nharmbasfns, Nmolorbs[spin])`` and is ready to be passed to
      :class:`stowfn.StoWfn`.

   .. method:: setup_stowfn()

      Construct and populate a :class:`stowfn.StoWfn` object from all
      previously computed attributes.

      Computes the nuclear repulsion energy from the ``Atomic Distances``
      matrix (for multi-atom systems), populates all geometry, basis, and
      orbital attributes on :attr:`sto`, and calls
      :meth:`stowfn.StoWfn.check_and_normalize` to validate the result.

   .. method:: apply_cusp_correction()

      Enforce nuclear cusp conditions on all MOs and write ``stowfn.data``.

      For each spin and each MO the cusp constraint violation **A c** is
      evaluated.  Any MO with a violation norm above ``1e-9`` is flagged and
      corrected according to :attr:`CUSP_METHOD`:

      .. list-table::
         :header-rows: 1
         :widths: 15 85

         * - Method
           - Action
         * - ``'enforce'``
           - Applies the cusp-enforcing matrix **M** to pin the coefficient
             of the highest-exponent s-type AO on each centre to the
             cusp-satisfying value while leaving all other coefficients
             unchanged.
         * - ``'project'``
           - Projects the coefficient vector onto the null space of **A**,
             removing all cusp-violating components.
         * - ``'none'``
           - No correction applied.

      When :attr:`PLOT_CUSPS` is ``True``, MO values and Laplacians are
      evaluated before and after correction for use by :meth:`plot_cusps`.
      After processing, ``stowfn.data`` is saved.

      :raises AssertionError: If any residual cusp violation exceeds
         ``1e-8`` after correction.

   .. method:: plot_cusps()

      Generate and save ``cusp_constraint.svg``.

      Does nothing when :attr:`PLOT_CUSPS` is ``False``.  Produces a
      ``2 × Natoms`` grid of subplots:

      * **Top row** — MO values along the z-axis through each nucleus
        (dashed = before cusp correction, solid = after).
      * **Bottom row** — local energy
        :math:`E_\mathrm{loc} = -\nabla^2\psi/(2\psi) - Z/|r|`
        along the same line, with y-limits set from the post-correction
        extrema.

      Only orbitals with a non-zero cusp violation (stored in
      ``self.fixed``) are plotted.  Sign normalization is applied so all
      MOs are positive at the midpoint of the z-axis segment.

      :raises ImportError: If ``matplotlib`` is not installed.

   .. method:: run()

      Execute the full ADF → CASINO conversion pipeline:

      1. :meth:`process_valence_basis`
      2. :meth:`process_core_basis`
      3. :meth:`process_shells`
      4. :meth:`process_coefficients`
      5. :meth:`process_core_orbitals`
      6. :meth:`finalize_coefficients`
      7. :meth:`setup_stowfn`
      8. :meth:`apply_cusp_correction` → writes ``stowfn.data``
      9. :meth:`plot_cusps` → writes ``cusp_constraint.svg`` (optional)


.. function:: main()

   Command-line entry point.

   Parses arguments with :mod:`argparse` and delegates to
   :meth:`ADFToStoWF.run`.


Command-line interface
----------------------

**Synopsis**

.. code-block:: console

   python adf2stowf.py [options]

**Options**

.. list-table::
   :header-rows: 1
   :widths: 35 12 53

   * - Flag
     - Default
     - Description
   * - ``--plot-cusps``
     - ``False``
     - Save ``cusp_constraint.svg`` with wavefunction values and local
       energies along the z-axis through each nucleus before and after
       cusp correction.
   * - ``--cusp-method {project,enforce,none}``
     - ``project``
     - Nuclear cusp correction strategy (see :meth:`apply_cusp_correction`).
   * - ``--all-orbitals``
     - ``False``
     - Include virtual (unoccupied) MOs in the output.
   * - ``--dump``
     - ``False``
     - Write a human-readable ``TAPE21.txt`` dump of all parsed ADF data.

**Input / output files**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``TAPE21.asc`` (input)
     - ASCII export of the ADF binary checkpoint, must be in the working
       directory.
   * - ``stowfn.data`` (output)
     - CASINO STO wavefunction file, always produced.
   * - ``TAPE21.txt`` (optional output)
     - Human-readable dump of parsed data (requires ``--dump``).
   * - ``cusp_constraint.svg`` (optional output)
     - Cusp diagnostic plot (requires ``--plot-cusps``).


Usage examples
--------------

**Minimal conversion (defaults)**

.. code-block:: console

   $ python adf2stowf.py

**Debug run — dump raw data and use projection cusp method**

.. code-block:: console

   $ python adf2stowf.py --dump --cusp-method=project

**Inspect cusps for all orbitals including virtuals**

.. code-block:: console

   $ python adf2stowf.py --plot-cusps --all-orbitals --cusp-method=enforce

**Skip cusp correction (diagnostic)**

.. code-block:: console

   $ python adf2stowf.py --cusp-method=none

**Programmatic use**

.. code-block:: python

   from adf2stowf import adf2stowf

   conv = adf2stowf.ADFToStoWF(
       plot_cusps=False,
       cusp_method='project',
       do_dump=False,
       only_occupied=True,
   )
   conv.run()

   # Access the resulting StoWfn object directly
   sto = conv.sto
   print('Number of MOs:', sto.num_molorbs)


.. _cart-to-harm:

Cartesian to spherical-harmonic transformation
----------------------------------------------

ADF stores MO coefficients in a *Cartesian* STO basis; CASINO requires *real
solid-harmonic* (spherical) basis functions.  For shells with angular momentum
ℓ ≥ 2, the number of Cartesian functions exceeds the number of spherical ones.
The extra components are themselves Slater orbitals (with the radial prefactor
raised by :math:`r^2`) and are represented **exactly** by appending a companion
shell — they are never discarded:

.. list-table::
   :header-rows: 1
   :widths: 12 20 20 48

   * - Shell
     - Cartesian fns
     - Spherical fns
     - Extra component(s) → companion shell
   * - d (ℓ=2)
     - 6
     - 5
     - 1 s-type: :math:`x^2 + y^2 + z^2` → +s shell
   * - f (ℓ=3)
     - 10
     - 7
     - 3 p-type: :math:`x(x^2+y^2+z^2)`,
       :math:`y(x^2+y^2+z^2)`, :math:`z(x^2+y^2+z^2)` → +p shell


Basis function ordering
~~~~~~~~~~~~~~~~~~~~~~~

The polynomial ordering within each shell type follows ``stowfdet.f90`` from
CASINO.  Spherical functions are labelled by magnetic quantum number *m*;
Cartesian monomials use exponent triples *(i, j, k)* with :math:`x^i y^j z^k`.

**d-shell** — Cartesian order: :math:`xy,\ xz,\ yz,\ x^2{-}y^2,\ 2z^2{-}x^2{-}y^2,\ x^2{+}y^2{+}z^2`

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - *m*
     - Real solid harmonic (unnormalized)
   * - :math:`-2`
     - :math:`xy`
   * - :math:`-1`
     - :math:`yz`
   * - :math:`0`
     - :math:`2z^2 - x^2 - y^2`
   * - :math:`+1`
     - :math:`xz`
   * - :math:`+2`
     - :math:`x^2 - y^2`
   * - (s-contam)
     - :math:`x^2 + y^2 + z^2`

**f-shell** — Cartesian order: :math:`x^3,\ x^2y,\ x^2z,\ xy^2,\ xyz,\ xz^2,\ y^3,\ y^2z,\ yz^2,\ z^3`

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - *m*
     - Real solid harmonic (unnormalized)
   * - :math:`0`
     - :math:`2z^3 - 3z(x^2+y^2)`
   * - :math:`+1`
     - :math:`4xz^2 - x(x^2+y^2)`
   * - :math:`-1`
     - :math:`4yz^2 - y(x^2+y^2)`
   * - :math:`+2`
     - :math:`(x^2-y^2)z`
   * - :math:`-2`
     - :math:`xyz`
   * - :math:`+3`
     - :math:`x^3 - 3xy^2`
   * - :math:`-3`
     - :math:`3x^2y - y^3`
   * - (p-contam x)
     - :math:`x(x^2+y^2+z^2)`
   * - (p-contam y)
     - :math:`y(x^2+y^2+z^2)`
   * - (p-contam z)
     - :math:`z(x^2+y^2+z^2)`


Matrix equations
~~~~~~~~~~~~~~~~

For each shell, the :attr:`harm2cart_map` matrix **M** (shape
:math:`N_\text{cart} \times N_\text{cart}`, *full square*) relates the
spherical-harmonic-plus-contamination polynomial coefficients
:math:`\mathbf{c}_\text{sph}` to Cartesian coefficients
:math:`\mathbf{c}_\text{cart}`:

.. math::

   \mathbf{c}_\text{cart} = \mathbf{M}\, \mathbf{c}_\text{sph}

The first :math:`N_\text{sph}` rows are the true spherical harmonics; the
remaining rows are the contamination components, which feed the appended
companion shell rather than being discarded.  Because the matrix is square and
invertible the mapping is exact (bijective).  Its inverse is stored in
:attr:`cart2harm_map`:

.. math::

   \mathbf{c}_\text{sph} = \mathbf{M}^{-1}\, \mathbf{c}_\text{cart}

**Normalisation.**  ADF MO coefficients refer to *individually normalised*
Cartesian monomials (the per-function ``bnorm`` factors stored in TAPE21),
whereas CASINO expects coefficients of *its own* normalised real harmonics
(:meth:`stowfn.StoWfn.get_norm`).  Within a d or f shell these norms differ
between components, so the bare polynomial inverse must be conjugated by both:

.. math::

   \mathbf{B} = \operatorname{diag}(1/\mathbf{n}_\text{CASINO})\;
                \mathbf{M}^{-1}\;
                \operatorname{diag}(\mathbf{n}_\text{ADF})

The shared radial factor cancels between the parent and contamination rows
(they have the same :math:`n = \ell + N + 1`), so the conjugation is purely
angular.  For s and p shells it reduces to the identity
(:math:`\mathbf{n}_\text{ADF} = \mathbf{n}_\text{CASINO}`, one component per
shell), which is why omitting it is invisible for s/p and for spherical atoms
but distorts molecular d/f-bearing orbitals by several mHa.

The per-shell blocks :math:`\mathbf{B}` are assembled into the global
block-diagonal :attr:`cart2harm_matrix` **C** (shape
:math:`N_\text{harm}^\text{total} \times N_\text{cart}^\text{total}`) over all
centres and shells, and applied to the MO coefficients:

.. math::

   \mathbf{c}_\text{sph}^\text{MO} = \mathbf{C}\, \mathbf{c}_\text{cart}^\text{MO}


.. _cusp-conditions:

Nuclear cusp conditions
-----------------------

Each MO must satisfy the cusp condition at every nucleus *c* with charge
*Z_c*:

.. math::

   \left.\frac{\partial \psi}{\partial r}\right|_{r=0} = -Z_c \, \psi(0)

This is encoded as a linear constraint **A c** = **0** on the AO coefficient
vector **c** (see :meth:`stowfn.StoWfn.cusp_constraint_matrix`).
The three ``--cusp-method`` strategies differ in how they restore this
condition when it is violated:

* **enforce** — adjusts only the coefficient of the highest-exponent s-type
  AO on each centre via the matrix **M** = **I** − **e**\ :sub:`fix`
  **A**:sup:`+` **A**, leaving all other coefficients unchanged.
* **project** — applies the null-space projector **Q Q**:sup:`T` to remove
  all cusp-violating components simultaneously.
* **none** — performs no correction; useful for diagnosis or when the
  original orbitals already satisfy the cusp.


Dependencies
------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Usage
   * - ``numpy``
     - All array arithmetic throughout the pipeline.
   * - ``scipy.linalg``
     - ``null_space`` for the ``project`` cusp-correction strategy
       (used by :meth:`stowfn.StoWfn.cusp_projection_matrix`).
   * - ``matplotlib``
     - Cusp diagnostic plots (imported lazily when ``--plot-cusps`` is used).
   * - ``argparse``
     - Command-line argument parsing in :func:`main`.
   * - ``adf2stowf.adfread``
     - :class:`adfread.AdfParser` for reading ``TAPE21.asc``.
   * - ``adf2stowf.stowfn``
     - :class:`stowfn.StoWfn` for constructing and writing ``stowfn.data``.
