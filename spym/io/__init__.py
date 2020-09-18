import os

def load(filename, scaling=True):
    ''' Import data from common SPM file formats.

    Currently supported file formats are:
        * NeXus (.nx, .nxs). Package nxarray is needed.
        * RHK (.sm4). Package rhksm4 is needed.
        * Omicron Scala (.par). Package omicronscala is needed.

    Args:
        filename: path to the SPM file.
        scaling: if True convert data to physical units (default), if False keep raw data.

    Returns:
        xarray Dataset with data and metadata.

    '''

    if filename.endswith(".nx") or filename.endswith(".NX") or filename.endswith(".nxs") or filename.endswith(".NXS"):
        try:
            import nxarray
        except ImportError:
            print("Error: nxarray package is needed to open .nx/.nxs files.")
            return None
        try:
            ds = nxarray.load(filename)
        except:
            print("Error: the file does not appear to be valid.")
            return None

    if filename.endswith(".par") or filename.endswith(".PAR"):
        try:
            import omicronscala
        except ImportError:
            print("Error: omicronscala package is needed to open .par files.")
            return None
        try:
            ds = omicronscala.to_dataset(filename, scaling=scaling)
        except:
            print("Error: the file does not appear to be valid.")
            return None

    if filename.endswith(".sm4") or filename.endswith(".SM4"):
        try:
            import rhksm4
        except ImportError:
            print("Error: rhksm4 package is needed to open .sm4 files.")
            ds = None
        try:
            ds = rhksm4.to_dataset(filename, scaling=scaling)
        except:
            print("Error: the file does not appear to be valid.")
            return None

    for dr in ds:
        ds[dr].attrs["filename"] = os.path.basename(filename)

    return ds
