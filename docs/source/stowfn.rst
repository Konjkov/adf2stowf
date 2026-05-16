stowfn — Slater-Type Orbital Wavefunction I/O and Evaluation
=============================================================

.. module:: stowfn
   :synopsis: Read, write and evaluate Slater-Type Orbital (STO) wavefunctions.

Overview
--------

This module provides the :class:`StoWfn` class for working with Slater-Type
Orbital (STO) wavefunction files in the format used by CASINO.  It handles:

* Parsing and serialising the structured plain-text ``.data`` file format;
* Evaluating atomic orbitals (AOs), molecular orbitals (MOs), their
  gradients and Laplacians at arbitrary positions in space;
* Computing normalization factors for the STO basis;
* Enforcing nuclear cusp conditions through projection and linear
  transformation matrices.

The module supports restricted and spin-unrestricted (alpha/beta) calculations
and basis sets up to **g**-type shells (angular momentum quantum number ℓ = 4).

Module-level constants
----------------------

.. data:: F2P_bool

   ``dict`` — Mapping from Fortran logical literals to Python booleans.

   .. code-block:: python

      {'.false.': False, '.true.': True}

.. data:: P2F_bool

   ``dict`` — Inverse mapping from Python booleans to Fortran logical literals.

   .. code-block:: python

      {False: '.false.', True: '.true.'}

.. data:: num_orbs_per_shelltype

   ``numpy.ndarray`` of shape ``(7,)`` — Number of angular basis functions
   (polynomials) for each shell-type code.

   ===========  =====  =====================================
   Shell code   Type   Number of functions
   ===========  =====  =====================================
   0            —      0 (unused placeholder)
   1            s      1
   2            sp     4  (one s + three p combined)
   3            p      3
   4            d      5
   5            f      7
   6            g      9
   ===========  =====  =====================================


StoWfn class
------------

.. class:: StoWfn(fname=None)

   Container and evaluator for a Slater-Type Orbital wavefunction.

   :param fname: Path to the wavefunction data file.  When supplied the file
      is parsed immediately.  When ``None`` an empty object is created via
      :meth:`initempty`.
   :type fname: str or None

   .. rubric:: Instance attributes (populated by :meth:`readfile`)

   **General / metadata**

   .. attribute:: title
      :type: str

      Title string from the first line of the file.

   .. attribute:: code
      :type: str

      Name of the quantum-chemistry code that generated the file.

   .. attribute:: periodicity
      :type: int

      Dimensionality / periodicity of the system (0 = molecule, 1/2/3 = periodic).

   .. attribute:: spin_unrestricted
      :type: bool

      ``True`` if the calculation is spin-unrestricted (separate alpha and
      beta orbitals).

   .. attribute:: nuclear_repulsion_energy
      :type: float

      Nuclear repulsion energy in atomic units per atom.

   .. attribute:: num_elec
      :type: int

      Total number of electrons.

   **Geometry**

   .. attribute:: num_atom
      :type: int

      Number of atoms in the system.

   .. attribute:: atompos
      :type: numpy.ndarray

      Atomic positions in bohr, shape ``(num_atom, 3)``.

   .. attribute:: atomnum
      :type: numpy.ndarray

      Atomic numbers (nuclear charges), shape ``(num_atom,)``.

   .. attribute:: atomcharge
      :type: numpy.ndarray

      Valence charges for each atom, shape ``(num_atom,)``.

   **Basis set**

   .. attribute:: num_centres
      :type: int

      Number of STO expansion centres.

   .. attribute:: centrepos
      :type: numpy.ndarray

      Cartesian coordinates of each centre in bohr, shape
      ``(num_centres, 3)``.

   .. attribute:: num_shells
      :type: int

      Total number of STO shells across all centres.

   .. attribute:: idx_first_shell_on_centre
      :type: numpy.ndarray

      Zero-based index of the first shell on each centre, plus a sentinel
      equal to ``num_shells`` appended at the end.
      Shape ``(num_centres + 1,)``.

   .. attribute:: shelltype
      :type: numpy.ndarray

      Shell-type code for every shell, shape ``(num_shells,)``.
      See :data:`num_orbs_per_shelltype` for the encoding.

   .. attribute:: order_r_in_shell
      :type: numpy.ndarray

      Power *N* of the radial prefactor ``r^N`` in each shell,
      shape ``(num_shells,)``.

   .. attribute:: zeta
      :type: numpy.ndarray

      Slater exponent ζ for each shell, shape ``(num_shells,)``.

   .. attribute:: num_atorbs
      :type: int

      Total number of atomic-orbital basis functions (AOs).

   .. attribute:: num_molorbs
      :type: numpy.ndarray

      Number of molecular orbitals per spin channel.
      Shape ``(1,)`` for restricted or ``(2,)`` for unrestricted calculations.

   **Derived attributes** (computed during file read / validation)

   .. attribute:: num_shells_on_centre
      :type: numpy.ndarray

      Number of shells on each centre, shape ``(num_centres,)``.

   .. attribute:: max_order_r_on_centre
      :type: numpy.ndarray

      Maximum radial order *N* present on each centre (at least 2),
      shape ``(num_centres,)``.

   .. attribute:: max_order_r
      :type: int

      Global maximum radial order *N* across all centres.

   .. attribute:: max_shell_type_on_centre
      :type: numpy.ndarray

      Highest shell-type code present on each centre, shape
      ``(num_centres,)``.  Used to gate computation of higher angular-
      momentum polynomials.

   .. attribute:: num_spins
      :type: int

      Number of spin channels (1 for restricted, 2 for unrestricted).

   **Orbital coefficients**

   .. attribute:: coeff
      :type: list of numpy.ndarray

      Expansion coefficients in the *un-normalized* AO basis.
      ``coeff[spin]`` has shape ``(num_molorbs[spin], num_atorbs)``.

   .. attribute:: coeff_norm
      :type: list of numpy.ndarray

      Expansion coefficients in the *normalized* AO basis, obtained by
      multiplying each column of ``coeff`` by the corresponding AO
      normalization factor from :meth:`get_norm`.
      ``coeff_norm[spin]`` has shape ``(num_molorbs[spin], num_atorbs)``.

   .. attribute:: footer
      :type: list of str

      Raw lines after the orbital-coefficient block, preserved verbatim on
      write-back.

   .. rubric:: Methods

   .. method:: initempty()

      Create an empty :class:`StoWfn` object without reading any file.
      Currently a no-op placeholder; subclasses may override it.

   .. method:: readfile(fname)

      Parse a wavefunction data file and populate all instance attributes.

      :param fname: Path to the file to read.
      :type fname: str
      :raises AssertionError: If the file structure is inconsistent (e.g.
         shell counts do not match basis function counts).

      The file is expected to follow a strict section-based format with
      labelled headers (``BASIC INFO``, ``GEOMETRY``, ``BASIS SET``,
      ``MULTIDETERMINANT INFORMATION``, ``ORBITAL COEFFICIENTS``).  Floating-
      point data blocks use fixed-width 20-character fields.

   .. method:: writefile(fname)

      Write all wavefunction data back to a file in the canonical format
      understood by :meth:`readfile`.

      :param fname: Destination file path.
      :type fname: str

   .. method:: check_and_normalize()

      Validate all attributes for type and shape consistency, recompute the
      derived attributes (:attr:`num_shells_on_centre`,
      :attr:`max_order_r_on_centre`, etc.) and refresh :attr:`coeff_norm`.

      :raises AssertionError: If any attribute has an unexpected type, dtype,
         or shape.

      Call this method after modifying attributes programmatically to ensure
      the object remains self-consistent before evaluation or file output.

   .. method:: read_molorbmods(fname='correlation.data')

      Read molecular orbital modifications from a CASINO-style
      ``correlation.data`` file.  The ``START MOLORBMODS`` / ``END MOLORBMODS``
      block is located and its ``START MOLECULAR ORBITAL COEFFICIENTS`` and
      ``START STO EXPONENT ZETAS`` sub-blocks are consumed (currently parsed
      but not applied; values are discarded).

      :param fname: Path to the correlation data file.
      :type fname: str

   .. method:: get_norm()

      Compute the normalization factor for every atomic orbital in the basis.

      The normalization constant for orbital *i* with angular polynomial *p*,
      radial order *N*, and exponent ζ is:

      .. math::

         \mathcal{N}_i = Y_p \cdot (2\zeta)^{n} \sqrt{\frac{2\zeta}{(2n)!}},
         \quad n = \ell_p + N + 1

      where :math:`Y_p` is the angular normalization factor for polynomial *p*
      (tabulated internally for s through g shells) and :math:`\ell_p` is the
      polynomial power (0 for s, 1 for p, 2 for d, …).

      :returns: Normalization factors, one per AO.
      :rtype: numpy.ndarray, shape ``(num_atorbs,)``

   .. method:: iter_atorbs()

      Iterate over all atomic orbitals in canonical order.

      :yields: ``(atorb, centre, nshell, N, pl)`` tuples where

         * ``atorb`` (*int*) — global AO index;
         * ``centre`` (*int*) — centre index;
         * ``nshell`` (*int*) — shell index within :attr:`shelltype`;
         * ``N`` (*int*) — radial order ``order_r_in_shell[nshell]``;
         * ``pl`` (*int*) — polynomial index within the shell.

   .. method:: eval_atorbs(pos)

      Evaluate all atomic basis functions at a set of points.

      The STO basis function for shell *s* at centre *c* evaluated at position
      **r** is:

      .. math::

         \phi_{s,p}(\mathbf{r}) = r^N \, P_p(\hat{\mathbf{r}}) \, e^{-\zeta |\mathbf{r} - \mathbf{R}_c|}

      where :math:`P_p` is the real solid-harmonic polynomial for component
      *p*, *N* is the radial order, and ζ is the Slater exponent.

      :param pos: Evaluation points in Cartesian coordinates (bohr).
      :type pos: numpy.ndarray, shape ``(3, num_points)``
      :returns: AO values at each point.
      :rtype: numpy.ndarray, shape ``(num_points, num_atorbs)``

      .. note::
         Shells whose contribution is below the numerical cutoff
         ``exp(-ζr) < exp(-746)`` are skipped to avoid floating-point
         underflow.

   .. method:: eval_molorbs(pos, spin=0)

      Evaluate molecular orbitals at a set of points.

      MOs are linear combinations of normalized AOs:

      .. math::

         \psi_j(\mathbf{r}) = \sum_i C_{ji}^{\text{norm}} \, \phi_i(\mathbf{r})

      :param pos: Evaluation points in Cartesian coordinates (bohr).
      :type pos: numpy.ndarray, shape ``(3, num_points)``
      :param spin: Spin channel index.  ``0`` = alpha (default);
         ``1`` = beta (only valid for unrestricted calculations).
      :type spin: int
      :returns: MO values at each point.
      :rtype: numpy.ndarray, shape ``(num_points, num_molorbs[spin])``

   .. method:: eval_molorb_derivs(pos, spin=0)

      Evaluate molecular orbitals together with their first derivatives
      (gradient) and second derivatives (Laplacian) at a set of points.

      :param pos: Evaluation points in Cartesian coordinates (bohr).
      :type pos: numpy.ndarray, shape ``(3, num_points)``
      :param spin: Spin channel index (see :meth:`eval_molorbs`).
      :type spin: int
      :returns: A three-element tuple ``(val, grad, lap)``:

         * ``val`` — orbital values,
           shape ``(num_points, num_molorbs[spin])``;
         * ``grad`` — orbital gradients ∂ψ/∂x, ∂ψ/∂y, ∂ψ/∂z,
           shape ``(3, num_points, num_molorbs[spin])``;
         * ``lap`` — orbital Laplacians ∇²ψ,
           shape ``(num_points, num_molorbs[spin])``.

      :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
      :raises NotImplementedError: If the basis contains g-type shells
         (ℓ = 4); gradients and Laplacians are not yet implemented for
         those angular functions.

   .. method:: cusp_constraint_matrix()

      Build the linear constraint matrix **A** that encodes the nuclear cusp
      conditions.

      For each nucleus *c* with charge *Z_c* and Slater exponent ζ, the cusp
      condition requires

      .. math::

         \left.\frac{\partial \psi}{\partial r}\right|_{r=0} = -Z_c \, \psi(0).

      The resulting matrix has one row per nucleus and one column per AO.
      For an s-type STO centred on nucleus *c*:

      * If *N* = 0: coefficient is :math:`\mathcal{N}(Z_c - \zeta)`;
      * If *N* = 1: coefficient is :math:`\mathcal{N}`.

      Contributions from AOs on other centres are evaluated at the nuclear
      position.

      :returns: Cusp constraint matrix.
      :rtype: numpy.ndarray, shape ``(num_centres, num_atorbs)``

   .. method:: cusp_projection_matrix()

      Construct the orthogonal projector **Q** onto the null space of the
      cusp constraint matrix **A**.

      Any orbital-coefficient vector in the image of **Q** satisfies
      **A** **x** = **0**, i.e., it imposes the cusp conditions exactly.

      .. math::

         \mathbf{Q} = \mathbf{V}_{\text{null}} \mathbf{V}_{\text{null}}^{\top}

      where the columns of :math:`\mathbf{V}_{\text{null}}` form an
      orthonormal basis for ``null_space(A)``.

      :returns: Cusp projection matrix.
      :rtype: numpy.ndarray, shape ``(num_atorbs, num_atorbs)``

   .. method:: cusp_fixed_atorbs()

      Identify the AO index that is "fixed" (pinned) by the cusp constraint
      on each centre.

      For each centre the s-type shell with the *largest* Slater exponent is
      selected as the representative AO; its coefficient is determined by the
      cusp condition rather than optimized freely.

      :returns: Array of fixed AO indices, one per centre.
      :rtype: numpy.ndarray, shape ``(num_centres,)``, dtype int

   .. method:: cusp_enforcing_matrix()

      Build the linear transformation **M** that maps *any* AO coefficient
      vector to one satisfying the cusp conditions.

      The matrix is constructed so that the "fixed" AO coefficients
      (see :meth:`cusp_fixed_atorbs`) are set by a least-squares inversion
      of the cusp constraint, leaving all other AOs unchanged:

      .. math::

         \mathbf{M} = \mathbf{I} - \mathbf{e}_{\text{fix}} \, \mathbf{A}^{+} \mathbf{A}

      where :math:`\mathbf{A}^{+}` is the pseudo-inverse restricted to the
      fixed AO columns.

      :returns: Cusp-enforcing transformation matrix.
      :rtype: numpy.ndarray, shape ``(num_atorbs, num_atorbs)``


Usage examples
--------------

**Reading and writing a wavefunction file**

.. code-block:: python

   from stowfn import StoWfn

   # Read wavefunction
   wfn = StoWfn('stowfn.data')

   # Inspect basis
   print(f'Number of atoms:   {wfn.num_atom}')
   print(f'Number of AOs:     {wfn.num_atorbs}')
   print(f'Number of MOs:     {wfn.num_molorbs}')
   print(f'Spin unrestricted: {wfn.spin_unrestricted}')

   # Round-trip to a new file
   wfn.writefile('stowfn_copy.data')

**Evaluating MOs and their derivatives on a grid**

.. code-block:: python

   import numpy as np
   from stowfn import StoWfn

   wfn = StoWfn('stowfn.data')

   # 100 random points in a 10 bohr cube
   rng = np.random.default_rng(42)
   pos = rng.uniform(-5, 5, size=(3, 100))

   # Orbital values (alpha spin)
   mo_vals = wfn.eval_molorbs(pos, spin=0)          # (100, n_mo)

   # Values + gradient + Laplacian
   val, grad, lap = wfn.eval_molorb_derivs(pos, spin=0)

**Applying cusp conditions**

.. code-block:: python

   from stowfn import StoWfn

   wfn = StoWfn('stowfn.data')

   # Projection matrix — project coefficients into the cusp-satisfying subspace
   Q = wfn.cusp_projection_matrix()          # (num_atorbs, num_atorbs)
   cusp_coeff = wfn.coeff[0] @ Q.T

   # Or use the enforcing matrix to pin fixed AOs
   M = wfn.cusp_enforcing_matrix()           # (num_atorbs, num_atorbs)
   enforced_coeff = wfn.coeff[0] @ M.T


File format
-----------

The wavefunction file is an ASCII text file divided into labelled sections.
Each section begins with a header line and a separator, followed by labelled
fields, one per line, with the value on the next line.  Floating-point arrays
use 20-character fixed-width columns (four per line); integer arrays use 10-
character columns (eight per line).

.. code-block:: none

   <title>

   BASIC INFO
   ----------
   Generated by:
    <code_name>
   Periodicity:
    <int>
   Spin unrestricted:
    <.true.|.false.>
   Nuclear repulsion energy (au/atom):
    <float>
   Number of electrons
    <int>

   GEOMETRY
   --------
   Number of atoms
    <int>
   Atomic positions (au)
    <3·num_atom floats, 3 per line>
   ...

   BASIS SET
   ---------
   Number of STO centres
    <int>
   ...

   MULTIDETERMINANT INFORMATION
   ----------------------------
   GS

   ORBITAL COEFFICIENTS (normalized AO)
   -------------------------------------
    <num_molorbs[0] · num_atorbs floats>
   [<num_molorbs[1] · num_atorbs floats>  (unrestricted only)]

   <footer lines>


Dependencies
------------

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Package
     - Usage
   * - ``numpy``
     - Array arithmetic; used throughout for all numerical data.
   * - ``scipy.linalg``
     - :func:`scipy.linalg.null_space` for :meth:`StoWfn.cusp_projection_matrix`.
   * - ``math``
     - :func:`math.factorial` and :func:`math.sqrt` for normalization.
