Installation
============

Requirements
------------

The script has been verified to work with:

* Python 3.9 - 3.11
* NumPy >= 1.24.4
* SciPy >= 1.13.1

For optional plotting of the cusp constraints:

* Matplotlib >= 3.9.0

System Dependencies
-------------------

To build the package from source, you must first install the required system dependencies. 
On Ubuntu/Debian-based systems, run:

.. code-block:: bash

   sudo apt update
   sudo apt install python3-dev python3-distutils python3-venv build-essential cmake

Installation Methods
--------------------

From PyPI
~~~~~~~~~

.. code-block:: bash

   pip install adf2stowf

From Source
~~~~~~~~~~~

.. code-block:: bash

   cd source-dir
   pip install .
