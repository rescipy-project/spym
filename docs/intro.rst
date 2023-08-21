Introduction
============

A python package for processing Scanning Tunneling Microscopy (STM) data from RHK, based on the `spym project <https://github.com/rescipy-project/spym>`_.
Source is available from `GitHub <https://github.com/zrbyte/rhkpy/>`_.


Known issues
------------

- For line spectra, topography (and probably all the other data) is loaded in the incorrect order. `coord_to_absolute()` doesn't work for the image part of line spectra.
- For the older version (`RHK_MinorVer = 5`) of the RHK software, the I(z) spectra seem to have the wrong `RHK_LineTypeName` field value. They show up as dI/dV spectra.


Installation
============

.. code-block::

	pip install rhkpy


Examples
=============

Take a look at the ``tutorial.ipynb`` Jupyter notebook in the `GitHub repository <https://github.com/zrbyte/rhkpy/>`_.

Below is a simple example.

.. code-block:: python
	
	import rhkpy

	# Example to come soon

