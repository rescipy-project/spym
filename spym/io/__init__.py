import os

def load(filename, scaling=True):
    ''' Import data from common SPM file formats.
    
    Currently supported file formats are:
        * NeXus (.nx, .nxs)
        * RHK (.sm4)
        * Omicron (.par)
    
    Args:
        filename: path to the SPM file
        scaling: if True convert data to physical units (default),
            if False keep raw data
    
    Returns:
        xarray Dataset with data and metadata.
    '''

    if filename.endswith(".nx") or filename.endswith(".nxs"):
        try:
            import nxarray
            ds = nxarray.load(filename)
        except:
            print("Error: nxarray package is needed to open .nx/.nxs files.")
            ds = None

    if filename.endswith(".par"):
        try:
            import omicronscala
            ds = omicronscala.to_dataset(filename, scaling=scaling)
        except:
            print("Error: omicronscala package is needed to open .par files.")
            ds = None

    if filename.endswith(".sm4"):
        try:
            import rhksm4
            ds = rhksm4.to_dataset(filename, scaling=scaling)
        except:
            print("Error: rhksm4 package is needed to open .sm4 files.")
            ds = None

    if ds is not None:
        for dr in ds:
            ds[dr].attrs["filename"] = os.path.basename(filename)

    return ds
