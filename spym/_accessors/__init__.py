import xarray as xr
import numpy as np

from .filters import SpymFilters
from .level import SpymLevel
from .plotting import SpymPlotting

@xr.register_dataarray_accessor("spym")
class _SpymDataArray:
    '''spym class extending xarray DataArray

    '''

    def __init__(self, xarray_dr):
        self._dr = xarray_dr

        self._bkg = np.zeros(self._dr.data.shape)
        self._mask = np.zeros(self._dr.data.shape, dtype=bool)

        # Initialize spym classes
        self.level = SpymLevel(self)
        self.filters = SpymFilters(self)
        self.plotting = SpymPlotting(self)

        # Expose some useful methods
        self.fixzero = self.level.fixzero
        self.plane = self.level.plane
        self.align = self.level.align
        self.destripe = self.filters.destripe
        self.plot = self.plotting.plot

    @property
    def background(self):
        ''' Background.

        '''
        return self._bkg

    @property
    def mask(self):
        ''' Mask.

        '''
        return self._mask

@xr.register_dataset_accessor("spym")
class _SpymDataset:
    '''spym class extending xarray Dataset

    '''

    def __init__(self, xarray_ds):
        self._ds = xarray_ds
