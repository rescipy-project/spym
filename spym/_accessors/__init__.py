import xarray as xr

from ..process import Filters
from ..process import Level
from ..plotting import Plotting

@xr.register_dataarray_accessor("spym")
class _SpymDataArray:
    '''spym class extending xarray DataArray

    '''

    def __init__(self, xarray_dr):
        self._dr = xarray_dr

        self._bkg = None
        self._mask = None

        # Initialize spym classes
        self.Filters = Filters(self)
        self.Level = Level(self)
        self.Plotting = Plotting(self)

        # Expose some useful methods
        self.fixzero = self.Level.fixzero
        self.plane = self.Level.plane
        self.align = self.Level.align
        self.destripe = self.Filters.destripe
        self.plot = self.Plotting.plot
        self.hvplot = self.Plotting.hvplot

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
