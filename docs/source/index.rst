ADF to CASINO converter
=======================

This package converts wave function data output from the **ADF** (Amsterdam
Density Functional) program into the ``stowfn.data`` input file for the
**CASINO** quantum Monte Carlo code.

ADF is the only major quantum chemistry program that uses
`Slater-Type Orbitals <https://en.wikipedia.org/wiki/Slater-type_orbital>`_
(STO) natively.  CASINO can use them directly by setting::

    atom_basis_type : slater-type

in the CASINO input file, which makes ADF+CASINO a powerful combination for
high-accuracy QMC calculations without the need for a GTO-to-STO re-expansion.

For general information about ADF, see https://www.scm.com/.
For CASINO, see https://casino.ph.utexas.edu/.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   adfread
   adf2stowf
   stowfn

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
