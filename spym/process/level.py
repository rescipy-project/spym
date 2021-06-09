import numpy as np

class Level():
    ''' Level.

    '''

    def __init__(self, spym_instance):
        self._spym = spym_instance

    def fixzero(self, **kwargs):
        ''' Add a constant to all the data to move the minimum (or the mean value) to zero.

        Args:
            to_mean: bool, optional. If true move mean value to zero, if false move mimimum to zero (default).

        '''

        self._spym._dr.data = fixzero(self._spym._dr.data, **kwargs)

    def plane(self, **kwargs):
        '''Corrects for sample tilting by subtraction of a plane.

        '''

        if not self._spym._dr.data.ndim == 2:
            print("The DataArray is not an image. Abort.")
            return

        self._spym._dr.data, self._spym._bkg = plane(self._spym._dr.data.astype(float), **kwargs)

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

        self._spym._dr.data, self._spym._bkg = align(self._spym._dr.data.astype(float), **kwargs)

def fixzero(image,
            to_mean=False):
    ''' Add a constant to all the data to move the minimum (or the mean value) to zero.

    Args:
        image: numpy array.
        to_mean: bool, optional. If true move mean value to zero, if false move mimimum to zero (default).

    Returns:
        numpy array.

    '''
    
    if to_mean:
        fixed = image - image.mean()
    else:
        fixed = image - image.min()

    return fixed

def plane(image):
    '''Corrects for image tilting by subtraction of a plane.

    Args:
        image: 2d numpy array.

    Returns:
        flattened image as 2d numpy array.

    '''

    bkg_x = _poly_bkg(image.mean(axis=0), 1)
    bkg_y = _poly_bkg(image.mean(axis=1), 1)

    bkg_xx = np.apply_along_axis(_fill, 1, image, bkg_x)
    bkg_yy = np.apply_along_axis(_fill, 0, image, bkg_y)

    bkg = bkg_xx + bkg_yy
    planned = image - bkg

    return planned, bkg

def align(image, baseline='mean', axis=1, poly_degree=2):
    '''Align rows.

    Args:
        image: 2d numpy array.
        baseline: defines how baselines are estimated; 'mean' (default), 'median', 'poly'.
        axis: axis along wich calculate the baselines.
        poly_degree: polnomial degree if baseline='poly'.

    Returns:
        corrected 2d numpy array.

    '''

    if baseline == 'mean':
        bkg = np.apply_along_axis(_mean_bkg, axis, image)
    elif baseline == 'median':
        bkg = np.apply_along_axis(_median_bkg, axis, image)
    elif baseline == 'poly':
        bkg = np.apply_along_axis(_poly_bkg, axis, image, poly_degree)

    aligned = image - bkg

    return aligned, bkg

def _mean_bkg(line):
    return np.full(line.shape[0], line.mean())

def _median_bkg(line):
    return np.full(line.shape[0], np.median(line))

def _poly_bkg(line, poly_degree):
    x = np.linspace(-.5, .5, line.shape[0])
    coefs = np.polyfit(x, line, poly_degree)
    return np.polyval(coefs, x)

def _fill(line, value):
    return value
