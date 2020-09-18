from spym.process import level as spym_level

class SpymLevel():
    ''' Level.

    '''

    def __init__(self, spym_instance):
        self._spym = spym_instance

    def fixzero(self, **kwargs):
        ''' Add a constant to all the data to move the minimum (or the mean value) to zero.

        Args:
            to_mean: bool, optional. If true move mean value to zero, if false move mimimum to zero (default).

        '''

        self._spym._dr.data = spym_level.fixzero(self._spym._dr.data, **kwargs)

    def plane(self, **kwargs):
        '''Corrects for sample tilting by subtraction of a plane.

        '''

        if not self._spym._dr.data.ndim == 2:
            print("The DataArray is not an image. Abort.")
            return

        self._spym._dr.data, self._spym._bkg = spym_level.plane(self._spym._dr.data.astype(float), **kwargs)

    def align(self, **kwargs):
        '''Align rows.

        Args:
            baseline: defines how baselines are estimated; 'mean' (default), 'median', 'poly'.
            axis: axis along wich calculate the baselines.
            poly_degree: polnomial degree if baseline='poly'.

        '''

        if not self._spym._dr.data.ndim == 2:
            print("The DataArray is not an image. Abort.")
            return

        self._spym._dr.data, self._spym._bkg = spym_level.align(self._spym._dr.data.astype(float), **kwargs)
