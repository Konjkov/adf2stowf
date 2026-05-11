Slater-Type Orbitals
====================

Definition
----------

Each Slater-type orbital (STO) centred on atom :math:`A` is defined as:

.. math::

    \chi_{nlm}(\mathbf{r}) = N_{nl}\, r^{n-1} e^{-\zeta r} Y_l^m(\hat{r})

where:

* :math:`n` — principal quantum number
* :math:`l` — angular momentum quantum number
* :math:`m` — magnetic quantum number
* :math:`\zeta` — orbital exponent (read from ``TAPE21``)
* :math:`Y_l^m(\hat{r})` — real spherical harmonics
* :math:`N_{nl}` — normalisation constant

Normalisation
-------------

The normalisation constant is chosen so that :math:`\langle\chi|\chi\rangle = 1`:

.. math::

    N_{nl} = \frac{(2\zeta)^{n+1/2}}{\sqrt{(2n)!}}

This follows from the radial integral:

.. math::

    \int_0^\infty r^{2(n-1)} e^{-2\zeta r} r^2\, dr
    = \frac{(2n)!}{(2\zeta)^{2n+1}}

Real Spherical Harmonics
------------------------

The real spherical harmonics :math:`Y_l^m` are related to the complex ones
:math:`\mathcal{Y}_l^m` by:

.. math::

    Y_l^m(\hat{r}) = \begin{cases}
        \dfrac{1}{\sqrt{2}}\bigl(\mathcal{Y}_l^{|m|} + (-1)^m \mathcal{Y}_l^{-|m|}\bigr) & m > 0 \\[6pt]
        \mathcal{Y}_l^0 & m = 0 \\[6pt]
        \dfrac{i}{\sqrt{2}}\bigl(\mathcal{Y}_l^{m} - (-1)^m \mathcal{Y}_l^{-m}\bigr) & m < 0
    \end{cases}
