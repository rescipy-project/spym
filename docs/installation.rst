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
For signal processing the `scipy <https://www.scipy.org/>`_ package is also required.
In the case you need to load and save data from and to NeXus/HDF5 through the `spym.load()` function the `nxarray <https://github.com/nxarray/nxarray>`_ package is needed.
