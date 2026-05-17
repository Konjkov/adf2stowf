adfread — ADF TAPE21.asc Parser
================================

.. module:: adfread
   :synopsis: Parse and dump ADF TAPE21.asc binary-export files.

Overview
--------

This module provides the :class:`AdfParser` class for reading ADF
(Amsterdam Density Functional) ``TAPE21.asc`` files — the ASCII export
format of the ADF binary checkpoint file ``TAPE21``.

The parser converts the structured plain-text representation into a nested
Python dictionary keyed by *group* and *variable name*, and can write a
human-readable dump of the parsed data.

The module can also be run directly as a command-line utility:

.. code-block:: console

   $ python adfread.py results.asc

This produces ``results.txt`` containing a formatted dump of all data.

Module-level constants
----------------------

.. data:: INT_FIELD_WIDTH

   ``int`` — Fixed column width (in characters) used for integer fields: ``12``.

.. data:: FLOAT_FIELD_WIDTH

   ``int`` — Fixed column width (in characters) used for floating-point
   fields: ``28``.

.. data:: STRING_BLOCK_SIZE

   ``int`` — Block size (in characters) for string entries: ``160``.


AdfParser class
---------------

.. class:: AdfParser(filename)

   Parser for ADF ``TAPE21.asc`` files.

   :param filename: Path to the ``.asc`` file to parse.
   :type filename: str or pathlib.Path

   .. rubric:: Instance attributes

   .. attribute:: filename
      :type: pathlib.Path

      Resolved path to the input file.

   .. attribute:: lines
      :type: list of str

      Raw lines of the file as loaded by :meth:`load`.  Empty until
      :meth:`load` or :meth:`parse` is called.

   .. attribute:: data
      :type: dict[str, dict[str, Any]]

      Nested dictionary of parsed values, populated by :meth:`parse`.
      The outer key is the *group* name; the inner key is the *variable*
      name; the value is a ``numpy.ndarray`` (integers or floats),
      a ``list`` (strings or booleans), depending on the type code
      in the file.

   .. rubric:: Static methods

   .. staticmethod:: _float_x(x)

      Convert an ADF-formatted floating-point string to a Python
      :class:`float`.

      Handles ADF-specific quirks such as bare exponents starting with
      ``'E'`` (missing mantissa) and negative-exponent notation.

      :param x: Raw field string from the file.
      :type x: str
      :returns: Parsed float value.
      :rtype: float

   .. staticmethod:: _split_n(s, n)

      Split a string into fixed-length chunks of exactly *n* characters,
      stripping surrounding whitespace from each chunk.

      :param s: Input string (typically one raw file line).
      :type s: str
      :param n: Chunk width in characters.
      :type n: int
      :returns: List of stripped substrings.
      :rtype: list of str

   .. staticmethod:: _int_x(s)

      Convert an ADF-formatted integer string to a Python :class:`int`.

      ADF uses the sentinel ``'**********'`` to represent integer overflow
      (values that do not fit in a signed 32-bit integer).  Such values
      are mapped to ``-(2**31)``.

      :param s: Raw field string from the file.
      :type s: str
      :returns: Parsed integer value or ``-(2**31)`` for the overflow marker.
      :rtype: int

   .. rubric:: Methods

   .. method:: load()

      Read the file into :attr:`lines`.

      Uses ``latin-1`` encoding to faithfully handle any byte values that
      may appear in ADF output.

      :raises SystemExit: If the file does not exist or is a directory;
         an error message is printed and the process exits with code 1.

   .. method:: parse()

      Parse all data from :attr:`lines` into :attr:`data`.

      Calls :meth:`load` automatically if :attr:`lines` is empty.

      **File structure understood by the parser:**

      Each variable is represented by three consecutive logical records:

      1. *Group name* — a section identifier string (e.g.
         ``'Geometry'``, ``'SCF'``).
      2. *Key name* — the variable name within the group.
      3. *Descriptor line* — three whitespace-separated integers:
         ``len1``, ``len2``, ``typ``.

         * ``len2`` — number of values to read.
         * ``typ`` — type code:

           .. list-table::
              :header-rows: 1
              :widths: 15 25 60

              * - Code
                - Type
                - Storage
              * - ``1``
                - Integer
                - ``numpy.ndarray``, dtype ``int``
              * - ``2``
                - Float
                - ``numpy.ndarray``, dtype ``float``
              * - ``3``
                - String
                - ``list`` of ``str``
              * - ``4``
                - Boolean
                - ``list`` of ``bool``

         * ``len1`` — when ``0``, one additional blank line follows the
           data block and is consumed.

      4. *Data block* — one or more lines holding ``len2`` values encoded
         in fixed-width fields (:data:`INT_FIELD_WIDTH`,
         :data:`FLOAT_FIELD_WIDTH`, or :data:`STRING_BLOCK_SIZE`).

      :returns: The populated :attr:`data` dictionary.
      :rtype: dict[str, dict[str, Any]]
      :raises ValueError: If an unknown type code (not 1–4) is encountered.
      :raises Exception: Re-raises any parsing exception after printing a
         context window of ±3 lines around the failing position.

   .. method:: write_dump(outfile)

      Write a human-readable text dump of :attr:`data` to *outfile*.

      Calls :meth:`parse` automatically if :attr:`data` is empty.

      The output format is:

      .. code-block:: none

         <group name>
           <key> = <value>
           <long_key> = {<count>}
               <value line 1>
               <value line 2>
               ...

      Multi-line values (those whose string representation contains a
      newline) are printed in the indented block form shown above.

      :param outfile: Destination file path.
      :type outfile: str or pathlib.Path

   .. rubric:: Private parsing helpers

   The following methods are used internally by :meth:`parse` and share
   the same calling convention:

   .. method:: _parse_integers(start, count)

      Read *count* integers beginning at line *start* of :attr:`lines`,
      consuming as many lines as needed (using :data:`INT_FIELD_WIDTH`
      column width).

      :param start: Starting line index.
      :type start: int
      :param count: Number of values to read.
      :type count: int
      :returns: ``(values, next_line_index)``
      :rtype: tuple(numpy.ndarray, int)

   .. method:: _parse_floats(start, count)

      Read *count* floats beginning at line *start*, using
      :data:`FLOAT_FIELD_WIDTH` column width.

      :param start: Starting line index.
      :type start: int
      :param count: Number of values to read.
      :type count: int
      :returns: ``(values, next_line_index)``
      :rtype: tuple(numpy.ndarray, int)

   .. method:: _parse_strings(start, count)

      Read *count* characters of raw text starting at line *start* and
      split into :data:`STRING_BLOCK_SIZE`-character blocks, stripping
      trailing whitespace from each block.

      :param start: Starting line index.
      :type start: int
      :param count: Number of characters to consume.
      :type count: int
      :returns: ``(values, next_line_index)``
      :rtype: tuple(list of str, int)

   .. method:: _parse_bools(start, count)

      Read *count* boolean values beginning at line *start*.  Each
      character on a line is decoded as ``'T'`` → ``True`` or
      ``'F'`` → ``False``.

      :param start: Starting line index.
      :type start: int
      :param count: Number of values to read.
      :type count: int
      :returns: ``(values, next_line_index)``
      :rtype: tuple(list of bool, int)

   .. method:: _parse_value(typ, start, count)

      Dispatch to the appropriate ``_parse_*`` helper based on *typ*.

      :param typ: Type code (1 = int, 2 = float, 3 = str, 4 = bool).
      :type typ: int
      :param start: Starting line index.
      :type start: int
      :param count: Number of values to read.
      :type count: int
      :returns: ``(values, next_line_index)``
      :rtype: tuple(Any, int)
      :raises ValueError: If *typ* is not in ``{1, 2, 3, 4}``.


Usage examples
--------------

**Parsing a file programmatically**

.. code-block:: python

   from adfread import AdfParser

   parser = AdfParser('TAPE21.asc')
   data = parser.parse()

   # Access a specific group and variable
   atom_coords = data['Geometry']['xyz']   # numpy.ndarray of floats
   atom_types  = data['Geometry']['atomtype']

   print(atom_coords.reshape(-1, 3))

**Writing a human-readable dump**

.. code-block:: python

   from adfread import AdfParser

   parser = AdfParser('TAPE21.asc')
   parser.write_dump('TAPE21.txt')

**Iterating over all groups and keys**

.. code-block:: python

   from adfread import AdfParser

   parser = AdfParser('TAPE21.asc')
   data = parser.parse()

   for group, variables in data.items():
       for key, value in variables.items():
           print(f'{group} / {key}: shape={getattr(value, "shape", len(value))}')

**Command-line usage**

.. code-block:: console

   $ python adfread.py TAPE21.asc
   # Produces TAPE21.txt

   # Error — wrong extension:
   $ python adfread.py results.dat
   Error: input file must end with .asc


Command-line interface
----------------------

When invoked as a script the module accepts a single positional argument:

.. code-block:: console

   python adfread.py <file.asc>

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Condition
     - Behaviour
   * - No argument supplied
     - Prints usage and exits with code 1.
   * - File extension is not ``.asc``
     - Prints an error and exits with code 1.
   * - File not found or is a directory
     - Prints an error and exits with code 1.
   * - Success
     - Writes ``<stem>.txt`` alongside the input file and exits with code 0.


Dependencies
------------

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Package
     - Usage
   * - ``numpy``
     - Storage for parsed integer and float arrays.
   * - ``re``
     - Regular-expression preprocessing of ADF float strings in
       :meth:`AdfParser._float_x`.
   * - ``pathlib``
     - Platform-independent path handling.
   * - ``sys``
     - Reading command-line arguments and controlled process exit.
   * - ``typing``
     - Type annotations (``Any``, ``Dict``, ``List``, ``Union``).
