Cartesian to Spherical Harmonic Conversion
==========================================

Background
----------

ADF computes molecular orbitals in a Cartesian Gaussian basis
:math:`\{x^a y^b z^c e^{-\alpha r^2}\}`, while CASINO requires orbitals
expressed in pure spherical harmonics. This conversion is performed for
every shell of angular momentum :math:`l`.

Transformation Matrix
---------------------

The pure spherical harmonic components :math:`\phi_m^l` are obtained from
the Cartesian components :math:`g_{abc}` via:

.. math::

    \phi_m^l = \sum_{\substack{a,b,c \\ a+b+c=l}} T_{m,abc}^l\; g_{abc}

where :math:`T_{m,abc}^l` are the Cartesian-to-spherical transformation
coefficients.

The number of Cartesian components for angular momentum :math:`l` is
:math:`(l+1)(l+2)/2`, while the number of pure spherical components is
:math:`2l+1`. For :math:`l \geq 2` there are more Cartesian than spherical
components — the difference is spanned by lower-:math:`l` polynomials
(contaminants).

Example: d-shell (:math:`l = 2`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------+---------------------+
| Cartesian (6)     | Spherical (5)       |
+===================+=====================+
| :math:`xx`        | :math:`m = -2`      |
+-------------------+---------------------+
| :math:`yy`        | :math:`m = -1`      |
+-------------------+---------------------+
| :math:`zz`        | :math:`m = 0`       |
+-------------------+---------------------+
| :math:`xy`        | :math:`m = +1`      |
+-------------------+---------------------+
| :math:`xz`        | :math:`m = +2`      |
+-------------------+---------------------+
| :math:`yz`        | *(contaminant)*     |
+-------------------+---------------------+

The redundant combination :math:`x^2 + y^2 + z^2` is an s-type contaminant
that must be removed.

Without ``--cart2harm-projection`` the code maps only the 5 pure spherical
components and implicitly assumes the contaminant is zero. A warning is
emitted if the actual contaminant amplitude is non-negligible.

Orthogonal Projection
---------------------

When ``--cart2harm-projection`` is enabled, an orthogonal projector
:math:`\mathbf{P}` is applied to the full MO coefficient matrix
:math:`\mathbf{C}`:

.. math::

    \tilde{\mathbf{C}} = \mathbf{P}\,\mathbf{C}

.. math::

    \mathbf{P} = \mathbf{T}^{\!\top}
    \!\left(\mathbf{T}\mathbf{T}^{\!\top}\right)^{-1}
    \mathbf{T}

where :math:`\mathbf{T}` is the transformation matrix defined above.
The projection discards all non-spherical components and retains exactly
the :math:`2l+1` physical degrees of freedom for each shell.

.. note::

    After projection the total energy may differ from the original
    Cartesian-basis value because energetically stabilizing but unphysical
    components are removed.
