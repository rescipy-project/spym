from ._sm4 import RHKsm4

def load(sm4file):
    '''This method load data and metadata from an RHK .sm4 file.
    
    Args:
        sm4file: the name of the .sm4 file to be loaded
    
    Returns:
        a container for the pages in the .sm4 file with their data and metadata
    
    Examples:
        f = rhksm4.load('/path/to/file.sm4') # load the file
        
        p0 = f[0] # assign first page in the file
        p0.label # returns page label name
        p0.data # returns page data as a numpy array
        p0.attrs # returns page metadata as a dictionary
    '''

    return RHKsm4(sm4file)

def to_dataset(sm4file, scaling=True):
    '''This method load an RHK .sm4 file into an xarray Dataset.
    
    The xarray package is required.
    
    Args:
        sm4file: the name of the .sm4 file to be loaded
        scaling: if True convert data to physical units (default),
            if False keep data in ADC units
    
    Returns:
        an xarray Dataset
    
    Examples:
        ds = rhksm4.to_dataset('/path/to/file.sm4')
        
        ds
        <xarray.Dataset>
        
        ds.IDxxxxx
        <xarray.DataArray>
    '''

    try:
        import xarray as xr
    except:
        print("Error: xarray package not found.")
        return

    f = load(sm4file)

    ds = xr.Dataset()
    for p in f:
        ds[p.label] = _to_datarr(p, scaling=scaling)

    return ds

def to_nexus(sm4file, filename=None, **kwargs):
    '''This method convert an RHK .sm4 file into a NeXus file.
    
    The nxarray package is required.
    
    Args:
        sm4file: the name of the .sm4 file to be converted
        filename: (optional) path of the NeXus file to be saved.
            If not provided, a NeXus file is saved in the same folder
            of the .sm4 file.
        **kwargs: any optional argument accepted by nexus NXdata.save() method
    
    Returns:
        nothing
    
    Examples:
        rhksm4.to_nexus('/path/to/file.sm4')
    '''

    try:
        import nxarray as nxr
    except:
        print("Error: nxarray package not found.")
        return

    if not filename:
        filename = sm4file._file.name

    ds = to_dataset(sm4file, scaling=False)
    ds.nxr.save(filename, **kwargs)

def _to_datarr(p, scaling):
    '''Create an xarray DataArray from an RHKPage
    '''

    import xarray as xr

    ## Create DataArray
    dr = xr.DataArray(p.data,
                      coords=p.coords,
                      attrs=p.attrs,
                      name=p.label)

    ## Set xarray/nexusformat attributes
    dr.attrs['long_name'] = dr.name.replace("_", " ")
    dr.attrs['units'] = dr.attrs['RHK_Zunits']

    dr.coords['x'].attrs['units'] = dr.attrs['RHK_Xunits']
    if 'y' in dr.coords:
        dr.coords['y'].attrs['units'] = dr.attrs['RHK_Yunits']

    ## Set additional nexusformat attributes
    dr.attrs['scaling_factor'] = dr.attrs['RHK_Zscale']
    dr.attrs['offset'] = dr.attrs['RHK_Zoffset']
    dr.attrs['start_time'] = dr.attrs['RHK_DateTime']
    dr.attrs['notes'] = dr.attrs['RHK_UserText']
    if dr.attrs['RHK_PageDataType'] == 1: # Line type
        dr.attrs['interpretation'] = 'spectrum'
    elif dr.attrs['RHK_PageDataType'] == 0: # Image type
        dr.attrs['interpretation'] = 'image'
    else:
        dr.attrs['interpretation'] = None

    ## Set additional NXstm nexusformat attributes
    dr.attrs['bias'] = dr.attrs['RHK_Bias']
    dr.attrs['bias_units'] = 'V'
    try:#TODO add support for correspondent PRM metadata
        dr.attrs['setpoint'] = dr.attrs['RHK_ZPI_SetPoint']
        dr.attrs['setpoint_units'] = dr.attrs['RHK_ZPI_SetPointUnit']
        if dr.attrs['RHK_ZPI_FeedbackType'] == 'Off':
            dr.attrs['feedback_active'] = False
        else:
            dr.attrs['feedback_active'] = True
            dr.attrs['feedback_pgain'] = dr.attrs['RHK_ZPI_ProportionalGain']
    except KeyError:
        dr.attrs['setpoint'] = 0
        dr.attrs['setpoint_units'] = ""
        dr.attrs['feedback_active'] = None
        dr.attrs['feedback_pgain'] = 0
    dr.attrs['scan_angle'] = dr.attrs['RHK_Angle']
    dr.attrs['time_per_point'] = dr.attrs['RHK_Period']

    dr.coords['x'].attrs['offset'] = dr.attrs['RHK_Xoffset']
    if 'y' in dr.coords:
        dr.coords['y'].attrs['offset'] = dr.attrs['RHK_Yoffset']

    # Set filename
    dr.attrs['filename'] = dr.attrs['RHK_FileName']
    dr.attrs.pop('RHK_FileName')

    # Set coordinates labels
    if dr.attrs['RHK_Xlabel'] == '':
        dr.coords['x'].attrs['long_name'] = 'x'
    else:
        dr.coords['x'].attrs['long_name'] = dr.attrs['RHK_Xlabel']
    if 'y' in dr.coords:
        if dr.attrs['RHK_PageDataType'] == 1:
            dr.coords['y'].attrs['long_name'] = 'Trace'
        else:
            dr.coords['y'].attrs['long_name'] = 'y'

    # Scale data to physical units
    if scaling:
        dr.data = dr.data.astype(float) * dr.attrs['scaling_factor'] + dr.attrs['offset']
        dr.attrs['scaling_factor'] = 1.0
        dr.attrs['offset'] = 0.0

    return dr
