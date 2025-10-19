ADF
===

This directory contains a converter script that takes wave function data
output from the ADF program and turns it into a input file for the CASINO
program.

For general information about the ADF program, see http://www.scm.com/


Requirements
============

The script has been verified to work with:

    Python 3.9.23
    NumPy 1.24.4
    SciPy 1.13.1

For optional plotting of the cusp constraints

    Matplotlib >=3.9.0

To build the package from source, you must first install the required system dependencies. On Ubuntu/Debian-based systems, run:

```bash
sudo apt update
sudo apt install python3-dev python3-distutils python3-venv build-essential cmake
cd source-dir
pip install .
```

or just `pip install adf2stowf`

Usage
=====

Run the adf program, e.g.

    adf < adf.in > adf.out

this should leave a binary file 'TAPE21' in the working directory.
Convert this binary file into ASCII format:

    dmpkf TAPE21 > TAPE21.asc

(the dmpkf utility is included with the ADF distribution)
Now run

    adf2stowf

in the same directory. This script will read 'TAPE21.asc' and write a file 'stowfn.data'.
This file can be used by CASINO setting the option

    atom_basis_type   : slater-type

in the CASINO input file.

The following command-line options are supported:
* `adf2stowf` — use default: `--cusp-method=enforce`
* `adf2stowf --plot-cusps` — enables cusps plotting (default: disabled)
* `adf2stowf --cusp-method=enforce` — apply transformation to active orbitals (default)
* `adf2stowf --cusp-method=project` — project out cusp-violating components
* `adf2stowf --cusp-method=none` — disable any cusp correction
* `adf2stowf --all-orbitals` — include also virtual orbitals (default: only occupied)
* `adf2stowf --dump` — generate a text dump of TAPE21.asc (default: no dump)
* `adf2stowf --cart2harm-projection` — enforce pure spherical harmonics via projection

You may see a warning like this during conversion:

    WARNING: cartesian to sperical conversion for spin 0, orb 0 violated by 0.00063567

This indicates that the original molecular orbitals (computed in Cartesian Gaussian basis) contain
non-spherical components — for example, s-type contamination (x²+y²+z²) in d- or f-shells. These
components violate angular momentum purity and are unphysical in a spherical harmonic representation.

You can eliminate this warning and enforce physically correct orbitals by enabling the `--cart2harm-projection` flag.
This flag applies an orthogonal projection that removes all constraint-violating components, ensuring your orbitals are
expressed strictly in pure spherical harmonics — as required CASINO code.

Without `--cart2harm-projection`, the code maps only the pure spherical components (e.g., 5 of 6 for d-shells), ignoring
the rest like the s-type contaminant (x²+y²+z²) and implicitly assumes they are zero.

❗ Important note on energy: Even after projection, and especially without it, the total energy may not match the original
Cartesian-basis energy because you’re removing unphysical (but energetically stabilizing) components without fully reconstructing
the wavefunction in the pure spherical basis.
