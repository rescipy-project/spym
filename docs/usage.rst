=====
Usage
=====

After installation you can import spym simply with:

.. code-block:: python
    
    import spym


Now the ``spym`` accessor will be available to xarray objects.
For example to apply the plane() function to an xarray DataArray:

.. code-block:: python
    
    dr = xarray.DataArray()
    dr.spym.plane()


Examples
========

The spym package is best integrated with xarray objects. Here is a typical usage for scanning tunneling microscopy data loaded into an xarray Dataset:

.. code-block:: python
    
    # Import the package
    import spym
    
    # Load the file and show the content
    f = spym.load("/path/to/a/supported/file")
    f
    
    # Select the channel of interest (e.g. Topography_Forward) and show its content
    tf = f.Topography_Forward
    tf
    
    # Align the rows
    tf.spym.align()
    
    # Make plane on the image
    tf.spym.plane()
    
    # Fix the minimum to zero
    tf.spym.fixzero()
    
    # Plot the image
    tf.spym.plot()


Many of the ``spym`` functions are also applicable directly to numpy arrays:

.. code-block:: python
    
    from spym.level import align
    
    aligned, background = align(my_array, baseline='median')
    

The documentation of each ``spym`` method can be accessed with the ``?`` syntax:

.. code-block:: python
    
    dr.spym.plane?

See the API Reference section for a list of all the methods available.


Supported file formats
======================

The ``spym`` package provides direct imports through the ``spym.load()`` function for a few file formats, at present:

* RHK SM4 *.sm4

* Omicron Scala *.par

* NeXus/HDF5 *.nxs (``nxarray`` package is needed)
