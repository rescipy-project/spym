=====
Usage
=====

After installation you can import spym simply with:

.. code-block:: python
    
    import spym


Now the ``spym`` accessor will be available to xarray objects, e.g.:

.. code-block:: python
    
    dr = xarray.DataArray()
    dr.spym.plane()


The documentation of each ``spym`` method can be accessed with:

.. code-block:: python
    
    dr.spym.plane?


.. 
    Examples
    ========

    Let's start by importing:
