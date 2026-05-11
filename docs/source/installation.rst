Installation
============

From PyPI
---------

The simplest way to install::

    pip install adf2stowf

From Source
-----------

To build the package from source, first install the required system dependencies.
On Ubuntu/Debian-based systems:

.. code-block:: bash

    sudo apt update
    sudo apt install python3-dev python3-distutils python3-venv build-essential cmake

Then install the package:

.. code-block:: bash

    cd source-dir
    pip install .

Virtual Environment (recommended)
----------------------------------

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate   # Linux/Mac
    # .venv\Scripts\activate    # Windows
    pip install adf2stowf

Dependencies
------------

Required:

* Python >= 3.9, < 3.12
* NumPy >= 1.24.4
* SciPy >= 1.13.1

Optional (for cusp plotting):

* Matplotlib >= 3.9.0
