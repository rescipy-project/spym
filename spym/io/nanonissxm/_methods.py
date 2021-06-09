from ._sxm import NANONISsxm

def load(sxmFile):
    '''This method load data and metadata associated to an Nanonis .sxm file.
    
    Args:
        sxmFile: the name of the .sxm file to be loaded
    
    Returns:
        a container for the channels in the .sxm file with their data and metadata
    
    Examples:
        f = nanonissxm.load('/path/to/file.sxm') # load the file
        
        ch0 = f[0] # assign first channel
        ch0.label # returns channel name label
        ch0.data # returns channel data as a numpy array
        ch0.attrs # returns channel metadata as a dictionary
    '''

    return NANONISsxm(sxmFile) 

def to_dataset(sxmFile, scaling=True):
    '''This method load an Nanonis .sxm file into an xarray Dataset.
    
    The xarray package is required.
    
    Args:
        sxmFile: the name of the .sxm file to be loaded
        scaling: if True convert data to physical units (default),
            if False keep data in ADC units
    
    Returns:
        an xarray Dataset
    
    Examples:
        ds = nanonissxm.to_dataset('/path/to/file.sxm')
        
        ds
        <xarray.Dataset>
        
        ds.Z_Forward
        <xarray.DataArray>
    '''

    try:
        import xarray as xr
    except:
        print("Error: xarray package not found.")
        return

    f = load(sxmFile)

    ds = xr.Dataset()
    for ch in f:
        ds[ch.label] = _to_datarr(ch, scaling=scaling)

    return ds

def to_nexus(sxmFile, filename=None, **kwargs):
    '''This method convert an Nanonis .sxm file into a NeXus file.
    
    The nxarray package is required.
    
    Args:
        sxmFile: the name of the .sxm file to be converted
        filename: (optional) path of the NeXus file to be saved.
            If not provided, a NeXus file is saved in the same folder
            of the .sxm file.
        **kwargs: any optional argument accepted by nexus NXdata.save() method
    
    Returns:
        nothing
    
    Examples:
        nanonissxm.to_nexus('/path/to/file.sxm')
    '''

    try:
        import nxarray as nxr
    except:
        print("Error: nxarray package not found.")
        return

    if not filename:
        import os
        filename = os.path.splitext(sxmFile)[0]

    ds = to_dataset(sxmFile, scaling=False)
    ds.nxr.save(filename, **kwargs)

def _to_datarr(ch, scaling):
    '''Create an xarray DataArray from an OMICRONchannel
    '''

    import xarray as xr

    ## Create DataArray
    dr = xr.DataArray(ch.data,
                      coords=ch.coords,
                      attrs=ch.attrs,
                      name=ch.label)

    ## Set xarray/nexusformat attributes
    dr.attrs['long_name'] = ch.label.replace("_", " ")
    dr.attrs['units'] = dr.attrs['PhysicalUnit']

    dr.coords['x'].attrs['units'] = 'nm'
    dr.coords['y'].attrs['units'] = 'nm'

    ## Set additional nexusformat attributes
    dr.attrs['scaling_factor'] = dr.attrs['Resolution']
    dr.attrs['offset'] = 0.0
    dr.attrs['start_time'] = dr.attrs['Timestamp']
    dr.attrs['notes'] = dr.attrs['Comment']

    ## Set additional NXstm nexusformat attributes
    dr.attrs['bias'] = dr.attrs['GapVoltage']
    dr.attrs['bias_units'] = 'V'
    dr.attrs['setpoint'] = dr.attrs['FeedbackSet']
    dr.attrs['setpoint_units'] = 'nA'
    dr.attrs['scan_angle'] = dr.attrs['ScanAngle']
    dr.attrs['feedback_active'] = True
    dr.attrs['feedback_pgain'] = dr.attrs['LoopGain']
    #dr.attrs['time_per_point'] = dr.attrs['TopographyTimeperPoint']

    dr.coords['x'].attrs['offset'] = dr.attrs['XOffset']
    dr.coords['x'].attrs['long_name'] = 'x'

    dr.coords['y'].attrs['offset'] = dr.attrs['YOffset']
    dr.coords['y'].attrs['long_name'] = 'y'

    # Scale data to physical units
    if scaling:
        dr.data = dr.data.astype(float) * dr.attrs['scaling_factor'] + dr.attrs['offset']
        dr.attrs['scaling_factor'] = 1.0
        dr.attrs['offset'] = 0.0

    return dr
