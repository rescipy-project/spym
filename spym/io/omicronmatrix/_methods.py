from ._matrix import OMICRONmatrix

def load(mtrxFile):
    '''This method load data and metadata associated to an Omicron MATRIX .*_mtrx file.
    
    Args:
        mtrxFile: the name of the .*_mtrx file to be loaded
    
    Returns:
        a container for the channels in the .*_mtrx file with their data and metadata
    
    Examples:
        f = omicronmatrix.load('/path/to/file.*_mtrx') # load the file
        
        ch0 = f[0] # assign first channel
        ch0.label # returns channel name label
        ch0.data # returns channel data as a numpy array
        ch0.attrs # returns channel metadata as a dictionary
    '''

    return OMICRONmatrix(mtrxFile) 

def to_dataset(mtrxFile):
    '''This method load an Omicron MATRIX .*_mtrx file into an xarray Dataset.
    
    The xarray package is required.
    
    Args:
        mtrxFile: the name of the .*_mtrx file to be loaded
    
    Returns:
        an xarray Dataset
    
    Examples:
        ds = omicronmatrix.to_dataset('/path/to/file.*_mtrx')
        
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

    f = load(mtrxFile)

    ds = xr.Dataset()
    for ch in f:
        ds[ch.label] = _to_datarr(ch)

    return ds

def to_nexus(mtrxFile, filename=None, **kwargs):
    '''This method convert an Omicron MATRIX .*_mtrx file into a NeXus file.
    
    The nxarray package is required.
    
    Args:
        mtrxFile: the name of the .*_mtrx file to be converted
        filename: (optional) path of the NeXus file to be saved.
            If not provided, a NeXus file is saved in the same folder
            of the .*_mtrx file.
        **kwargs: any optional argument accepted by nexus NXdata.save() method
    
    Returns:
        nothing
    
    Examples:
        omicronmatrix.to_nexus('/path/to/file.*_mtrx')
    '''

    try:
        import nxarray as nxr
    except:
        print("Error: nxarray package not found.")
        return

    if not filename:
        import os
        filename = os.path.splitext(mtrxFile)[0]#TODO

    ds = to_dataset(mtrxFile)
    ds.nxr.save(filename, **kwargs)

def _to_datarr(ch):
    '''Create an xarray DataArray from an OMICRONchannel
    '''

    import xarray as xr

    ## Create DataArray
    dr = xr.DataArray(ch.data,
                      coords=ch.coords,
                      attrs=ch.attrs,
                      name=ch.label)

    '''
    ## Set xarray/nexusformat attributes
    dr.attrs['long_name'] = ch.label.replace("_", " ")
    dr.attrs['units'] = dr.attrs['PhysicalUnit']

    dr.coords['x'].attrs['units'] = 'nm'
    dr.coords['y'].attrs['units'] = 'nm'

    ## Set additional nexusformat attributes
    dr.attrs['scaling_factor'] = 1.0
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
    '''

    return dr
