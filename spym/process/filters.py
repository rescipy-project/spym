import numpy as np
from scipy import ndimage

def gauss(image, 
          size=3):
    ''' Apply Gaussian smoothing filter

    Args:
        image: numpy array
        size: size of the filter in pixels
    
    Returns:
        filtered numpy array
    '''

    sigma = size * 0.42466

    return ndimage.filters.gaussian_filter(image, sigma)

def median(image,
           size=3):
    ''' Apply median smoothing filter

    Args:
        image: numpy array
        size: size of the filter in pixels
    
    Returns:
        filtered numpy array
    '''

    return ndimage.filters.median_filter(image, size=size)

def mean(image,
         size=3):
    ''' Apply mean smoothing filter

    Args:
        image: numpy array
        size: size of the filter in pixels
    
    Returns:
        filtered numpy array
    '''

    return ndimage.filters.uniform_filter(image, size=size)

def sharpen(image,
            size=3,
            alpha=30):
    ''' Apply a sharpening filter

    Args:
        image: numpy array
        size: size of the filter in pixels
        alpha: weight
    
    Returns:
        filtered numpy array
    '''

    blurred = ndimage.gaussian_filter(image, size)
    filter_blurred = ndimage.gaussian_filter(blurred, 1)
    sharpened = blurred + alpha * (blurred - filter_blurred)

    return sharpened

def destripe(image,
             min_lenght=20,
             hard_threshold = 0.4,
             soft_threshold = 0.2,
             only_mask=False):
    ''' Find and remove scan stripes by averaging neighbourhood lines

    Args:
        image: 2d numpy array
        min_lenght: only scars that are as long or longer than this value (in pixels) will be marked
        hard_threshold: the minimum difference of the value from the neighbouring upper and lower lines to be considered a defect.
        soft_threshold: values differing at least this much do not form defects themselves,
        but they are attached to defects obtained from the hard threshold if they touch one.
        only_mask: wheter returning just the mask (True) or also the corrected data (False, default).

    Returns:
        destriped 2d array
    '''

    # Calculate line differences
    d1 = np.diff(image, axis=0)
    diff1 = np.empty(image.shape)
    diff1[-1] = d1[-1]
    diff1[:-1] = d1

    d2 = np.diff(diff1, axis=0)
    diff2 = np.empty(image.shape)
    diff2[0] = d2[0]
    diff2[1:] = d2

    # Normalize line differences
    diff = -2*(diff2 - diff2.mean())/(diff2.max()-diff2.min())

    # Calculate masks for hard and soft thresholds
    m_hard = abs(diff) > hard_threshold
    m_soft = abs(diff) > soft_threshold

    # Opening (erosion+dilation) of the masks
    m_hard = ndimage.binary_opening(m_hard, structure=np.ones((1,min_lenght), dtype=bool))
    m_soft = ndimage.binary_opening(m_soft, structure=np.ones((1,3*min_lenght), dtype=bool))

    # Addition of hard and soft mask
    mask = ndimage.binary_opening(m_soft+m_hard, structure=np.ones((1,min_lenght), dtype=bool))

    # Filter masked values
    image_masked = np.ma.array(image, mask = mask, fill_value=np.NaN)
    filt = ndimage.uniform_filter(image_masked.data, size=(3, 1))

    if not only_mask:
        filtered = image*np.invert(mask) + filt*mask

    return filtered, mask
