Introduction
============

A python package for processing Scanning Tunneling Microscopy (STM) data from RHK, based on the `spym project <https://github.com/rescipy-project/spym>`_.
Source is available from `GitHub <https://github.com/zrbyte/rhkpy/>`_.

To make the most of rhkpy, also consult the `xarray documentation <https://docs.xarray.dev/en/latest/index.html>`_ and check out `plotting examples <https://holoviews.org/reference/index.html>`_, with HoloViews. The HoloViews website has `examples <https://holoviews.org/user_guide/Plotting_with_Bokeh.html>`_ on how to customize plots, as well as tutorial notebooks.


Known issues
------------

- For line spectra, topography (and probably all the other data) is loaded in the incorrect order. The topography data of line spectra should not be used at the moment.
- The method: `coord_to_absolute()` doesn't work for the image part of line spectra.
- For the older version (`RHK_MinorVer = 5`) of the RHK software, the I(z) spectra seem to have the wrong `RHK_LineTypeName` field value. They show up as dI/dV spectra.

Notes
------------

The "forward" scan direction in rhkpy is the "right" scan direction, when the file is opened in Gwyddion.


Installation
============

.. code-block::

	pip install rhkpy

	# to upgrade to a new version, use
	pip install --upgrade rhkpy

Specific example on a Windows machine
-------------------------------------

- `Download <https://winpython.github.io>`_ WinPython (3.xx), from GitHub or Sourceforge.
- After installing, in the WinPython directory, start the WindowsPowerShell.exe. Here you can run the "pip" commands.
- Start the "Jupyter Lab.exe" to run Jupyter notebooks.

Setting the notebook directory on Windows
-----------------------------------------

- In PowerShell, run this command to generate a Jupyter config file: "jupyter notebook --generate-config"
- The config file will be located in the sub-directory of WinPython: "python-3.xx.x.amd64/etc/jupyter"
- In the config file look for the option: "c.NotebookApp.notebook_dir"
- Change this to the desired directory.


Examples
=============

Take a look at the ``tutorial.ipynb`` Jupyter notebook in the `GitHub repository <https://github.com/zrbyte/rhkpy/>`_.

Below is a simple example.

.. code-block:: python
	
	import rhkpy

	# Load an sm4 file
	data = rhkpy.rhkdata('filename.sm4')

	# "quick plot" of the data
	data.qplot()

	# make thumbnails of the sm4 files in the current working directory
	rhkpy.genthumbs()


