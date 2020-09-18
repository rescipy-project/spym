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

Spym extends and is best integrated with the xarray package, so it is highly recommended to install it:

* `xarray <http://xarray.pydata.org>`_

Nevertheless, many spym functions can be used directly on numpy arrays (see the Usage section for more information).
In the case you need to load data from some proprietary format through the `spym.load()` function, the following packages are also needed:

* `rhksm4 <https://gitlab.com/rhksm4/rhksm4>`_

* `omicronscala <https://gitlab.com/mpanighel/omicronscala>`_
