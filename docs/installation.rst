============
Installation
============

You can install spym in your system through pip:

.. code-block:: bash

    $ pip install spym

.. 
    or, if you are using conda:

    .. code-block:: bash

        $ conda config --add channels conda-forge rescipy
        $ conda install spym


Prerequisites
=============

Spym extends and is best integrated with the `xarray <http://xarray.pydata.org>`_ package, which is required for the installation.

Nevertheless, many spym functions can be used directly on numpy arrays (see the Usage section for more information).
In the case you need to load data  through the `spym.load()` function from NeXus/HDF5 files or some supported proprietary formats, the following packages could be needed (see the Usage section for more information):

* `nxarray <https://github.com/nxarray/nxarray>`_

* `rhksm4 <https://gitlab.com/rhksm4/rhksm4>`_

* `omicronscala <https://gitlab.com/mpanighel/omicronscala>`_
