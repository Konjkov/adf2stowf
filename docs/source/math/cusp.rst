Cusp Correction
===============

The Cusp Condition
------------------

Near nucleus :math:`A` with atomic number :math:`Z_A` the electronic
wave function must satisfy the Kato cusp condition:

.. math::

    \left.\frac{\partial \psi}{\partial r_A}\right|_{r_A=0}
    = -Z_A\, \psi(0)

Molecular orbitals computed in a Gaussian basis generally violate this
condition because Gaussian functions have zero derivative at the nucleus.
STOs satisfy it by construction, but the MO coefficients transferred from
ADF may still introduce violations.

Three methods are available via ``--cusp-method``.

Method: ``none``
----------------

No correction is applied. Use only if the basis already satisfies the
cusp condition or the violation is negligible.

Method: ``enforce``
-------------------

An s-type STO correction term is added to each occupied orbital
:math:`\phi_i`:

.. math::

    \tilde{\phi}_i(\mathbf{r})
    = \phi_i(\mathbf{r})
    + \sum_A c_{iA}\, s_A(\mathbf{r})

where :math:`s_A` is an s-type STO centred on atom :math:`A`.
The coefficients :math:`c_{iA}` are determined by enforcing the cusp
condition exactly at each nucleus:

.. math::

    \left.\frac{d\tilde{\phi}_i}{dr_A}\right|_{r_A=0}
    + Z_A\,\tilde{\phi}_i(0) = 0

This gives a linear system for :math:`c_{iA}` which is solved per orbital
per atom. The method preserves the orbital norm to first order.

Method: ``project``
-------------------

The cusp-violating component of each orbital is removed by orthogonal
projection. For each atom :math:`A` the cusp-violation vector
:math:`\mathbf{b}_A` is computed and a projector is formed:

.. math::

    \hat{\Pi}_A = \mathbf{1} - \mathbf{v}_A \mathbf{v}_A^\dagger,
    \qquad
    \mathbf{v}_A = \frac{\mathbf{b}_A}{\|\mathbf{b}_A\|}

The corrected orbital is:

.. math::

    \tilde{\phi}_i = \prod_A \hat{\Pi}_A\, \phi_i

Unlike ``enforce``, this method does not add new basis functions but
slightly changes the orbital norm.
