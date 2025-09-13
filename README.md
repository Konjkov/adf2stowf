ADF
===

This directory contains a converter script that takes wave function data
output from the ADF program and turns it into a input file for the CASINO
program.

For general information about the ADF program, see http://www.scm.com/

For help and further information about this script, please contact the author:
    Norbert Nemec <Norbert@Nemec-online.de>


Requirements
============

The script has been verified to work with:

    Python 3.9.23
    NumPy 1.24.4
    scipy-weave 0.19.0

For optional plotting of the cusp constraints

    Matplotlib 3.9+


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
