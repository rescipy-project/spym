from spym.process import filters as spym_filters

class SpymFilters():
    ''' Filters.
    
    '''

    def __init__(self, spym_instance):
        self._spym = spym_instance

    def gauss(self, in_place=True, **kwargs):
        ''' Apply Gaussian smoothing filter.

        Args:
            size: size of the filter in pixels.
            in_place: if True the function is applied to the DataArray (default),
                if False a new DataArray is returned.
        '''

        if in_place:
            self._spym._dr.data = spym_filters.gauss(self._spym._dr.data, **kwargs)
        else:
            return self._spym._dr.copy(deep=True, data=spym_filters.gauss(self._spym._dr.data, **kwargs))

    def median(self, in_place=True, **kwargs):
        ''' Apply median smoothing filter.

        Args:
            size: size of the filter in pixels.
            in_place: if True the function is applied to the DataArray (default),
                if False a new DataArray is returned.
        '''

        if in_place:
            self._spym._dr.data = spym_filters.median(self._spym._dr.data, **kwargs)
        else:
            return self._spym._dr.copy(deep=True, data=spym_filters.median(self._spym._dr.data, **kwargs))

    def mean(self, in_place=True, **kwargs):
        ''' Apply mean smoothing filter.

        Args:
            size: size of the filter in pixels.
            in_place: if True the function is applied to the DataArray (default),
                if False a new DataArray is returned.
        '''

        if in_place:
            self._spym._dr.data = spym_filters.mean(self._spym._dr.data, **kwargs)
        else:
            return self._spym._dr.copy(deep=True, data=spym_filters.mean(self._spym._dr.data, **kwargs))

    def sharpen(self, in_place=True, **kwargs):
        ''' Apply a sharpening filter.

        Args:
            size: size of the filter in pixels.
            alpha: weight.
            in_place: if True the function is applied to the DataArray (default),
                if False a new DataArray is returned.
        '''

        if in_place:
            self._spym._dr.data = spym_filters.sharpen(self._spym._dr.data, **kwargs)
        else:
            return self._spym._dr.copy(deep=True, data=spym_filters.sharpen(self._spym._dr.data, **kwargs))

    def destripe(self, in_place=True, **kwargs):
        ''' Find and remove scan stripes by averaging neighbourhood lines.

        Args:
            min_lenght: only scars that are as long or longer than this value (in pixels) will be marked.
            hard_threshold: the minimum difference of the value from the neighbouring upper and lower lines to be considered a defect.
            soft_threshold: values differing at least this much do not form defects themselves,
                but they are attached to defects obtained from the hard threshold if they touch one.
            only_mask: wheter returning just the mask (True) or also the corrected data (False, default).
            in_place: if True the function is applied to the DataArray (default),
                if False a new DataArray is returned.
        '''

        if not self._spym._dr.data.ndim == 2:
            print("The DataArray is not an image. Abort.")
            return

        if in_place:
            self._spym._dr.data, self._spym._mask = spym_filters.destripe(self._spym._dr.data.astype(float), **kwargs)
        else:
            return self._spym._dr.copy(deep=True, data=spym_filters.destripe(self._spym._dr.data.astype(float), **kwargs)[0])
