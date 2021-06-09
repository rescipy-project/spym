import numpy as np
from scipy import ndimage

class Filters():
    ''' Filters.

    '''

    def __init__(self, spym_instance):
        self._spym = spym_instance

    def gauss(self, **kwargs):
        ''' Apply Gaussian smoothing filter.

        Args:
            size: size of the filter in pixels.

        '''

        self._spym._dr.data = gauss(self._spym._dr.data, **kwargs)

    def median(self, **kwargs):
        ''' Apply median smoothing filter.

        Args:
            size: size of the filter in pixels.

        '''

        self._spym._dr.data = median(self._spym._dr.data, **kwargs)

    def mean(self, **kwargs):
        ''' Apply mean smoothing filter.

        Args:
            size: size of the filter in pixels.

        '''

        self._spym._dr.data = mean(self._spym._dr.data, **kwargs)

    def sharpen(self, **kwargs):
        ''' Apply a sharpening filter.

        Args:
            size: size of the filter in pixels.
            alpha: weight.

        '''

        self._spym._dr.data = sharpen(self._spym._dr.data, **kwargs)

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

        self._spym._dr.data, self._spym._mask = destripe(self._spym._dr.data.astype(float), **kwargs)

def gauss(image, 
          size=3):
    ''' Apply Gaussian smoothing filter.

    Args:
        image: numpy array.
        size: size of the filter in pixels.

    Returns:
        filtered numpy array.

    '''

    sigma = size * 0.42466

    return ndimage.filters.gaussian_filter(image, sigma)

def median(image,
           size=3):
    ''' Apply median smoothing filter.

    Args:
        image: numpy array.
        size: size of the filter in pixels.

    Returns:
        filtered numpy array.

    '''

    return ndimage.filters.median_filter(image, size=size)

def mean(image,
         size=3):
    ''' Apply mean smoothing filter.

    Args:
        image: numpy array.
        size: size of the filter in pixels.

    Returns:
        filtered numpy array.

    '''

    return ndimage.filters.uniform_filter(image, size=size)

def sharpen(image,
            size=3,
            alpha=30):
    ''' Apply a sharpening filter.

    Args:
        image: numpy array.
        size: size of the filter in pixels.
        alpha: weight.

    Returns:
        filtered numpy array.

    '''

    blurred = ndimage.gaussian_filter(image, size)
    filter_blurred = ndimage.gaussian_filter(blurred, 1)
    sharpened = blurred + alpha * (blurred - filter_blurred)

    return sharpened

def destripe(image,
             min_length = 20,
             hard_threshold = 0.4,
             soft_threshold = 0.2,
             sign = 'positive',
             rel_threshold = None):
    ''' Find and remove scan stripes by averaging neighbourhood lines.

    Args:
        image: 2d numpy array.
        min_length: only scars that are as long or longer than this value (in pixels) will be marked.
        hard_threshold: the minimum difference of the value from the neighbouring upper and lower lines to be considered a defect.
        soft_threshold: values differing at least this much do not form defects themselves, but they are attached to defects obtained from the hard threshold if they touch one.
        sign: whether mark stripes with positive values, negative values or both.
        rel_threshold: the minimum difference of the value from the neighbouring upper and lower lines to be considered a defect (in physical values). Overwrite hard_threshold.

    Returns:
        destriped 2d array.

    '''

    # Normalize image
    rng = (image.max() - image.min()) / 2
    n_image = (image - image.mean())/rng

    # Calculate positive line differences
    d_pos = np.diff(n_image.clip(0, None), axis=0)
    np.clip(d_pos, 0, None, out=d_pos)
    diff_pos = np.empty(image.shape)
    diff_pos[0] = d_pos[0]
    diff_pos[1:] = d_pos

    # Calculate negative line differences
    d_neg = np.diff(n_image.clip(None, 0), axis=0)
    np.clip(d_neg, None, 0, out=d_neg)
    diff_neg = np.empty(image.shape)
    diff_neg[0] = d_neg[0]
    diff_neg[1:] = d_neg
    
    # Calculate physical threshold
    if rel_threshold:
        hard_threshold = rel_threshold*rng

    # Calculate masks for hard and soft thresholds
    m_hard_pos = False
    m_soft_pos = False
    m_hard_neg = False
    m_soft_neg = False
    if sign in ['positive', 'both']:
        m_hard_pos = diff_pos > hard_threshold
        m_soft_pos = diff_pos > soft_threshold
    if sign in ['negative', 'both']:
        m_hard_neg = diff_neg < -hard_threshold
        m_soft_neg = diff_neg < -soft_threshold

    # Opening (erosion+dilation) of the masks
    m_hard = ndimage.binary_opening(m_hard_pos+m_hard_neg, structure=np.ones((1,min_length), dtype=bool))
    m_soft = ndimage.binary_opening(m_soft_pos+m_soft_neg, structure=np.ones((1,2*min_length), dtype=bool))

    # Addition of hard and soft mask
    mask = ndimage.binary_opening(m_soft+m_hard, structure=np.ones((1,min_length), dtype=bool))

    # Filter masked values
    image_masked = np.ma.array(image, mask = mask, fill_value=np.NaN)
    filt = ndimage.uniform_filter(image_masked.data, size=(3, 1))

    filtered = image*np.invert(mask) + filt*mask

    return filtered, mask
