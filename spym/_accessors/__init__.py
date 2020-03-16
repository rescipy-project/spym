import xarray as xr
import numpy as np

@xr.register_dataarray_accessor("spym")
class _spymDataArray:
    '''spym class extending xarray DataArray
    '''

    def __init__(self, xarray_dr):
        self._dr = xarray_dr

        self._bkg = np.zeros(self._dr.data.shape)
        self._mask = np.zeros(self._dr.data.shape, dtype=bool)

    @property
    def background(self):
        return self._bkg

    @property
    def mask(self):
        return self._mask

    def fixzero(self, **kwargs):
        ''' Add a constant to all the data to move the minimum (or the mean value) to zero.
        
        Args:
            to_mean: bool, optional. If true move mean value to zero, if false move mimimum to zero (default).
        '''

        from spym.process.level import fixzero
        self._dr.data = fixzero(self._dr.data, **kwargs)

    def plane(self, **kwargs):
        '''Corrects for sample tilting by subtraction of a plane.
        
        '''

        if not self._dr.data.ndim == 2:
            print("The DataArray is not an image. Abort.")
            return
        from spym.process.level import plane
        self._dr.data, self._bkg = plane(self._dr.data.astype(float), **kwargs)

    def align(self, **kwargs):
        '''Align rows.
        
        Args:
            baseline: defines how baselines are estimated; 'median' (default), 'mean', 'poly'.
            axis: axis along wich calculate the baselines.
            poly_degree: polnomial degree if baseline='poly'.
        '''

        if not self._dr.data.ndim == 2:
            print("The DataArray is not an image. Abort.")
            return
        from spym.process.level import align
        self._dr.data, self._bkg = align(self._dr.data.astype(float), **kwargs)

    def destripe(self, **kwargs):
        ''' Find and remove scan stripes by averaging neighbourhood lines

        Args:
            min_lenght: only scars that are as long or longer than this value (in pixels) will be marked
            hard_threshold: the minimum difference of the value from the neighbouring upper and lower lines to be considered a defect.
            soft_threshold: values differing at least this much do not form defects themselves, but they are attached to defects obtained from the hard threshold if they touch one.
            only_mask: wheter returning just the mask in spym.mask (True) or also the corrected data (False, default).
        '''

        if not self._dr.data.ndim == 2:
            print("The DataArray is not an image. Abort.")
            return
        from spym.process.filters import destripe
        self._dr.data, self._mask = destripe(self._dr.data.astype(float), **kwargs)

@xr.register_dataset_accessor("spym")
class _spymDataset:
    '''spym class extending xarray Dataset
    '''

    def __init__(self, xarray_ds):
        self._ds = xarray_ds
