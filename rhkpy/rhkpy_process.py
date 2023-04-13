import xarray as xr
import pylab as pl
import numpy as np
from scipy import ndimage

def plot_spec_position(stmdata_object):
	xx = stmdata_object.image.coords['x'].data
	yy = stmdata_object.image.coords['y'].data
	dx = (xx[1] - xx[0])/2.
	dy = (yy[1] - yy[0])/2.
	ext = [xx[0] - dx, xx[-1] + dx, yy[0] - dy, yy[-1] + dy]

	scanangle = stmdata_object.image.attrs['scan angle']

	rotated_topo = ndimage.rotate(
		stmdata_object.image.isel(scandir = 0)['topography'].data,
		scanangle,
		reshape=False,
		mode='constant',
		cval=-0.1
		)

	pl.imshow(rotated_topo, extent=ext)
	pl.scatter(stmdata_object.spectra['x'].data, stmdata_object.spectra['y'].data,
		marker = '.',
		alpha = 0.5,
		color = 'r'
		)


