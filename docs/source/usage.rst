Usage
=====

Basic Workflow
--------------

1. Run the ADF program to generate wave function data:

.. code-block:: bash

   adf < adf.in > adf.out

This should leave a binary file ``TAPE21`` in the working directory.

2. Convert the binary file into ASCII format:

.. code-block:: bash

   dmpkf TAPE21 > TAPE21.asc

The ``dmpkf`` utility is included with the ADF distribution.

3. Run the converter:

.. code-block:: bash

   adf2stowf

This script will read ``TAPE21.asc`` and write a file ``stowfn.data``.

4. Use the output with CASINO by setting the option:

.. code-block:: text

   atom_basis_type   : slater-type

Command-Line Options
--------------------

The following command-line options are supported:

``adf2stowf``
    Use default: ``--cusp-method=enforce``

``adf2stowf --plot-cusps``
    Enable cusps plotting (default: disabled)

``adf2stowf --cusp-method=enforce``
    Apply transformation to active orbitals (default)

``adf2stowf --cusp-method=project``
    Project out cusp-violating components

``adf2stowf --cusp-method=none``
    Disable any cusp correction

``adf2stowf --all-orbitals``
    Include also virtual orbitals (default: only occupied)

``adf2stowf --dump``
    Generate a text dump of TAPE21.asc (default: no dump)

``adf2stowf --cart2harm-projection``
    Enforce pure spherical harmonics via projection

Important Notes
---------------

Cartesian to Spherical Conversion Warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may see a warning like this during conversion:

.. code-block:: text

   WARNING: cartesian to sperical conversion for spin 0, orb 0 violated by 0.00063567

This indicates that the original molecular orbitals (computed in Cartesian Gaussian basis) contain
non-spherical components — for example, s-type contamination (x²+y²+z²) in d- or f-shells. These
components violate angular momentum purity and are unphysical in a spherical harmonic representation.

You can eliminate this warning and enforce physically correct orbitals by enabling the 
``--cart2harm-projection`` flag. This flag applies an orthogonal projection that removes all 
constraint-violating components, ensuring your orbitals are expressed strictly in pure spherical 
harmonics — as required CASINO code.

Without ``--cart2harm-projection``, the code maps only the pure spherical components 
(e.g., 5 of 6 for d-shells), ignoring the rest like the s-type contaminant (x²+y²+z²) 
and implicitly assumes they are zero.

.. warning::

   Even after projection, and especially without it, the total energy may not match the original
   Cartesian-basis energy because you're removing unphysical (but energetically stabilizing) 
   components without fully reconstructing the wavefunction in the pure spherical basis.

Examples
--------

Basic conversion with default settings:

.. code-block:: bash

   adf2stowf

Include virtual orbitals and enable cusp plotting:

.. code-block:: bash

   adf2stowf --all-orbitals --plot-cusps

Use projection method for cusp enforcement with spherical harmonics projection:

.. code-block:: bash

   adf2stowf --cusp-method=project --cart2harm-projection
