from spym.process import filters as spym_filters

class SpymFilters():
    ''' Filters.

    '''

    def __init__(self, spym_instance):
        self._spym = spym_instance

    def gauss(self, **kwargs):
        ''' Apply Gaussian smoothing filter.

        Args:
            size: size of the filter in pixels.

        '''

        self._spym._dr.data = spym_filters.gauss(self._spym._dr.data, **kwargs)

    def median(self, **kwargs):
        ''' Apply median smoothing filter.

        Args:
            size: size of the filter in pixels.

        '''

        self._spym._dr.data = spym_filters.median(self._spym._dr.data, **kwargs)

    def mean(self, **kwargs):
        ''' Apply mean smoothing filter.

        Args:
            size: size of the filter in pixels.

        '''

        self._spym._dr.data = spym_filters.mean(self._spym._dr.data, **kwargs)

    def sharpen(self, **kwargs):
        ''' Apply a sharpening filter.

        Args:
            size: size of the filter in pixels.
            alpha: weight.

        '''

        self._spym._dr.data = spym_filters.sharpen(self._spym._dr.data, **kwargs)

    def destripe(self, **kwargs):
        ''' Find and remove scan stripes by averaging neighbourhood lines.

        Args:
            min_length: only scars that are as long or longer than this value (in pixels) will be marked.
            hard_threshold: the minimum difference of the value from the neighbouring upper and lower lines to be considered a defect.
            soft_threshold: values differing at least this much do not form defects themselves, but they are attached to defects obtained from the hard threshold if they touch one.
            sign: whether mark stripes with positive values, negative values or both.
            rel_threshold: the minimum difference of the value from the neighbouring upper and lower lines to be considered a defect (in physical values). Overwrite hard_threshold.

        Returns:
            destriped 2d array.

        '''

        if not self._spym._dr.data.ndim == 2:
            print("The DataArray is not an image. Abort.")
            return

        self._spym._dr.data, self._spym._mask = spym_filters.destripe(self._spym._dr.data.astype(float), **kwargs)
