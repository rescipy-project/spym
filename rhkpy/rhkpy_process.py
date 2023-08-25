import matplotlib.pyplot as pl
import numpy as np
import xarray as xr
import copy, glob, re
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import ndimage
import hvplot.xarray
import holoviews as hv

## internal functions


## functions ------------------------------------------------------

def coord_to_absolute(xrobj):
	"""Takes as input the :class:`rhkdata.image` variable of an :class:`~rhkpy.rhkpy_loader.rhkdata` instance.
	Returns a new :py:mod:`xarray` instance, with the coordinates updated to reflect the abolute tip position. This includes X, Y offset and rotation.

	:param xrobj: :py:mod:`xarray` image variable of an :class:`~rhkpy.rhkpy_loader.rhkdata` object
	:type xrobj: :py:mod:`xarray` Dataset
	
	:return: :py:mod:`xarray` :class:`rhkdata.image` instance, with the same data and metadata as the input and the coordinates shifted to absolute tip positions.
	:rtype: :py:mod:`xarray` Dataset

	:Example:
		
		.. code-block:: python

			import rhkpy

			m = rhkpy.rhkdata('didv map.sm4')

			# Take the `rhkdata` instance (image or map): `m`,
			# and convert the image coordinates to absolute values
			m_abs_image = rhkpy.coord_to_absolute(m.image)

			# coordinates of the instance `m`
			# We can see it runs from 0 to 100 nm
			print(m.image.x.min().data, m.image.x.max().data)
			0.0 100.0

			# check the same corrdinate for the new `m_abs`
			print(m_abs_image.x.min().data, m_abs_image.x.max().data)
			-877.0008892433623 -741.0633876547834

			# we can see it's now shows the exact tip position
			# the image is also rotated, as the "scan angle" attribute shows
			m_abs_image.attrs['scan angle']
			30.0
			
			# plot the rotated and offset image
			m_abs_image.topography.sel(scandir = 'forward').plot()
	"""	
	# the xrobj passed to the function should always be and image
	if 'topography' not in xrobj.data_vars:
		print('Wrong xarray type. The data needs to be an `image`, not `spectra`')
		return
	if xrobj.attrs['datatype'] == 'line':
		print('Sorry, linespectra are not supportet yet.')
		return
	
	# get scan angle
	scangle = xrobj.attrs['scan angle']*np.pi/180 # in radians

	# Get the numpy data
	datatopofw = xrobj['topography'].sel(scandir = 'forward').data
	datatopobw = xrobj['topography'].sel(scandir = 'backward').data
	datacurrentfw = xrobj['current'].sel(scandir = 'forward').data
	datacurrentbw = xrobj['current'].sel(scandir = 'backward').data
	dataliafw = xrobj['lia'].sel(scandir = 'forward').data
	dataliabw = xrobj['lia'].sel(scandir = 'backward').data

	# rotate the data by the scan angle. Need to have negative degrees, because ndimage rotates clockwise
	rotatedtopofw = ndimage.rotate(
		datatopofw,
		-scangle*180/np.pi, # needs to be in degrees
		reshape = True, # expand
		mode = 'constant',
		cval = np.nan
		)
	rotatedtopobw = ndimage.rotate(
		datatopobw,
		-scangle*180/np.pi, # needs to be in degrees
		reshape = True, # expand
		mode = 'constant',
		cval = np.nan
		)
	rotatedcurrentfw = ndimage.rotate(
		datacurrentfw,
		-scangle*180/np.pi, # needs to be in degrees
		reshape = True, # expand
		mode = 'constant',
		cval = np.nan
		)
	rotatedcurrentbw = ndimage.rotate(
		datacurrentbw,
		-scangle*180/np.pi, # needs to be in degrees
		reshape = True, # expand
		mode = 'constant',
		cval = np.nan
		)
	rotatedliafw = ndimage.rotate(
		dataliafw,
		-scangle*180/np.pi, # needs to be in degrees
		reshape = True, # expand
		mode = 'constant',
		cval = np.nan
		)
	rotatedliabw = ndimage.rotate(
		dataliabw,
		-scangle*180/np.pi, # needs to be in degrees
		reshape = True, # expand
		mode = 'constant',
		cval = np.nan
		)

	# Create new coordinates for the rotated data
	# size of a pixel in nm
	pixelsizex = np.abs(xrobj.x.data[1] - xrobj.x.data[0])
	pixelsizey = np.abs(xrobj.y.data[1] - xrobj.y.data[0])
	
	# Get the sizes of the x and y coordinates
	xlen = np.abs(xrobj.x.data[-1] - xrobj.x.data[0]) + pixelsizex # need to add half pixel size twice (on both sides)
	ylen = np.abs(xrobj.y.data[-1] - xrobj.y.data[0]) + pixelsizey
	
	# This gives you the new "bounding box size" of the rotated image
	# newxlen = np.abs(xlen * np.sin(scangle)) + np.abs(ylen * np.sin(np.pi/2 - scangle))
	# newylen = np.abs(xlen * np.cos(scangle)) + np.abs(ylen * np.cos(np.pi/2 - scangle))
	newxlen = rotatedtopofw.shape[0] * pixelsizex
	newylen = rotatedtopofw.shape[1] * pixelsizey

	# new coordinate length
	# placing the zero in the middle of the image
	newxx = np.linspace(-newxlen/2, newxlen/2, num = rotatedtopofw.shape[0])
	newyy = np.linspace(-newylen/2, newylen/2, num = rotatedtopofw.shape[1])
	# new pixel size due to rotation
	newpixelsizex = np.abs(newxx[1] - newxx[0])
	newpixelsizey = np.abs(newyy[1] - newyy[0])

	# correction to the offet of the image
	# In the RHK Rev software, the offsets shown in the software refer to the bottom - left corner
	# of the image. This does NOT include the rotation. For the proper shift of the image
	# coordinates including rotation this has to be taken into account
	diag = np.sqrt(xlen**2 + ylen**2)
	offx = diag * np.sin(scangle/2) * np.cos(scangle/2 - np.pi/4) + diag * np.cos(np.pi/4 + scangle)/2 - pixelsizex
	offy = diag * np.sin(scangle/2) * np.sin(scangle/2 - np.pi/4) + diag * np.sin(np.pi/4 + scangle)/2 - pixelsizey

	# make a new instance of the object, where we will change the coordinates
	xrobj_abscoord = xr.Dataset(
		data_vars = dict(
			topography = (['y', 'x', 'scandir'], np.stack((rotatedtopofw, rotatedtopobw), axis=-1)),
			current = (['y', 'x', 'scandir'], np.stack((rotatedcurrentfw, rotatedcurrentbw), axis=-1)),
			lia = (['y', 'x', 'scandir'], np.stack((rotatedliafw, rotatedliabw), axis=-1))
			),
		coords = dict(
			x = newxx + xrobj.attrs['xoffset'] + offx,
			y = newyy + xrobj.attrs['yoffset'] + offy,
			scandir = np.array(['forward', 'backward'])
			)
		)
	# copy attributes from original dataset and modify them accordingly
	xrobj_abscoord.attrs = xrobj.attrs.copy()
	for c in xrobj.coords:
		xrobj_abscoord.coords[c].attrs = xrobj.coords[c].attrs.copy()
	for d in xrobj.data_vars:
		xrobj_abscoord[d].attrs = xrobj[d].attrs.copy()
	# append a note to the coordinate x, y attributes
	xrobj_abscoord.attrs['comment'] = 'absolute coordinates'
	xrobj_abscoord.coords['x'].attrs['note'] += 'absolute coordinates\n'
	xrobj_abscoord.coords['y'].attrs['note'] += 'absolute coordinates\n'

	return xrobj_abscoord


def gaussian(x, x0 = 0, ampl = 2, width = 0.1, offset = 0):
	"""Gaussian function. Width and amplitude parameters have the same meaning as for :func:`lorentz`.

	:param x: values for the x coordinate
	:type x: float, :py:mod:`numpy` array, etc.
	:param x0: shift along the `x` corrdinate
	:type x0: float
	:param ampl: amplitude of the peak
	:type ampl: float
	:param width: FWHM of the peak
	:type width: float
	:param offset: offset along the function value
	:type offset: float

	:return: values of a Gaussian function
	:rtype: float, :py:mod:`numpy` array, etc.
	"""	
	# using the FWHM for the width
	return offset + ampl * np.exp(-2*np.log(2)*(x - x0)**2 / (width**2))


def lorentz(x, x0 = 0, ampl = 2, width = 0.1, offset = 0):
	"""Single Lorentz function

	:param x: values for the x coordinate
	:type x: float, :py:mod:`numpy` array, etc.
	:param x0: `x` corrdinate
	:type x0: float
	:param ampl: amplitude of the peak
	:type ampl: float
	:param width: FWHM of the peak
	:type width: float
	:param offset: offset along the function value
	:type offset: float

	:return: values of a single Lorentz function
	:rtype: float, :py:mod:`numpy` array, etc.

	.. note::
		The area of the peak can be given by:
		
		.. code-block:: python

			area = np.pi * amplitude * width / 2
	
	"""
	area = np.pi * ampl * width / 2
	return offset + (2/np.pi) * (area * width) / (4*(x - x0)**2 + width**2)

def gaussian2(x, x01 = -5, ampl1 = 1, width1 = 0.1, x02 = 5, ampl2 = 1, width2 = 0.1, offset = 0):
	"""Double Gaussian function

	:param x: values for the x coordinate
	:type x: float, :py:mod:`numpy` array, etc.
	:param x01: position of the peak, defaults to -5
	:type x01: float, optional
	:param ampl1: amplitude of the peak, defaults to 1
	:type ampl1: float, optional
	:param width1: width of the peak, defaults to 10
	:type width1: float, optional
	:param x02: position of the peak, defaults to 5
	:type x02: float, optional
	:param ampl2: amplitude of the peak, defaults to 1
	:type ampl2: float, optional
	:param width2: width of the peak, defaults to 10
	:type width2: float, optional
	:param offset: offset, defaults to 0
	:type offset: float, optional

	:return: values of a double Gaussian function
	:rtype: float, :py:mod:`numpy` array, etc.
	"""
	return offset + ampl1 * np.exp(-2*np.log(2)*(x - x01)**2 / (width1**2)) + ampl2 * np.exp(-2*np.log(2)*(x - x02)**2 / (width2**2))

def polynomial_fit(order, x_data, y_data):
	"""Polinomial fit to `x_data`, `y_data`

	:param order: order of the polinomial to be fit
	:type order: int
	:param x_data: x coordinate of the data
	:type x_data: :py:mod:`numpy` array
	:param y_data: y coordinate of the data
	:type y_data: :py:mod:`numpy` array

	:return: coefficients of the polinomial ``coeff``, as used by :py:mod:`numpy.polyval`, covariance matrix ``covar``, as returned by :py:mod:`scipy.optimize.curve_fit`
	:rtype: tuple: (:py:mod:`numpy` array, :py:mod:`numpy` array)

	"""    
	# Define polynomial function of given order
	def poly_func(x, *coeffs):
		y = np.polyval(coeffs, x)
		return y

	# Initial guess for the coefficients is all ones
	guess = np.ones(order + 1)

	# Use curve_fit to find best fit parameters
	coeff, covar = curve_fit(poly_func, x_data, y_data, p0 = guess)

	return coeff, covar

def bgsubtract(x_data, y_data, polyorder = 1, toplot = False, fitmask = None, hmin = 0.5, hmax = 10000, wmin = 1.5, wmax = 20, prom = 2, exclusion_factor = 3, peak_pos = None):
	"""Takes the ``x_data`` and ``y_data`` and automatically finds peaks, using :py:mod:`scipy.find_peaks`.
	These peaks are then used to define the areas of the background signal (``y_data``).
	In the areas with the peaks removed, the background is fitted by a polynomial of order given by the optional argument: ``polyorder``.
	The fit is performed by :py:mod:`scipy.optimize.curve_fit`.
	The function returns the ``y_data`` values with the background removed, the background polinomial values themselves and the coefficients of the background fit results, as used by :py:mod:`numpy.polyval`.

	In cases, where the automatic peak find is not functioning as expected, one can pass the values in ``x_data``, at which peaks appear.
	In this case, the ``wmin`` option determines the width of all peaks.

	If a ``fitmask`` is supplied for fitting, the fitmask is not calculated and only a polynomial fit is performed.
	This can decrease the runtime.

	:param x_data: variable of the data (typically bias voltage)
	:type x_data: :py:mod:`numpy` array
	:param y_data: data values (typically dI/dV)
	:type y_data: :py:mod:`numpy` array
	:param polyorder: order of polynomial used to fit the background, defaults to 3
	:type polyorder: int, optional
	:param toplot: if `True` a plot of: the fit, the background used and positions of the peaks is shown, defaults to False
	:type toplot: bool, optional
	:param fitmask: Fitmask to be used for polynomial fitting.
	:type fitmask: :py:mod:`numpy` array
	:param hmin: minimum height of the peaks passed to :py:mod:`scipy.signal.find_peaks`, defaults to 50
	:type hmin: float, optional
	:param hmax: maximum height of the peaks passed to :py:mod:`scipy.signal.find_peaks`, defaults to 10000
	:type hmax: float, optional
	:param wmin: minimum width of the peaks, passed to :py:mod:`scipy.signal.find_peaks`, defaults to 4
	:type wmin: float, optional
	:param wmax: maximum width of the peaks passed to :py:mod:`scipy.signal.find_peaks`, defaults to 60
	:type wmax: float, optional
	:param prom: prominence of the peaks, passed to :py:mod:`scipy.signal.find_peaks`, defaults to 10
	:type prom: float, optional
	:param exclusion_factor: this parameter multiplies the width of the peaks found by :py:mod:`scipy.signal.find_peaks`, or specified by ``wmin`` if the peak positions are passed by hand, using ``peak_pos``, defaults to 6
	:type exclusion_factor: float, optional
	:param peak_pos: list of the peak positions in ``x_data`` values used for exclusion, defaults to None
	:type peak_pos: list of floats, optional

	:return: ``y_data_nobg``, ``bg_values``, ``coeff``, ``params_used_at_run``, ``mask``, ``covar``
	:rtype: tuple: (:py:mod:`numpy` array, :py:mod:`numpy` array, :py:mod:`numpy` array, dictionary, :py:mod:`numpy` array, :py:mod:`numpy` array)

	* ``y_data_nobg``: data values, with the background subtracted,
	* ``bg_values``: the polynomial values of the fit, at the ``x_data`` positions,
	* ``coeff``: coefficients of the polynomial fit, as used by: :py:mod:`numpy.polyval`,
	* ``params_used_at_run``: parameters used at runtime
	* ``mask``: the calculated fitmask
	* ``covar``: covariance of the fit parameters

	.. note::
		Using the option: ``peak_pos``, a ``wmin*exclusion_factor/2`` region (measured in datapoints) on both sides of the peaks is excluded from the background fit.
		If automatic peak finding is used, the exclusion area is calculated in a similar way, but the width of the individual peaks are used, as determined by :py:mod:`scipy.signal.find_peaks`.

	"""
	# if a mask is passed to the function, don't calculate the peak positions
	if fitmask is None:
		if peak_pos is None:
			# Find the peaks with specified minimum height and width
			# re_height sets the width from the maximum at which value the width is evaluated
			peak_properties = find_peaks(y_data, height = (hmin, hmax), width = (wmin, wmax), prominence = prom, rel_height = 0.5)

			# Find the indices of the peaks
			peak_indices = peak_properties[0]

			# Get the properties of the peaks
			peak_info = peak_properties[1]
		else:
			# Use the provided peak positions
			peak_indices = []
			for peak_position in peak_pos:
				# Find the index of the closest data point to the peak position
				closest_index = np.argmin(np.abs(x_data - peak_position))
				peak_indices.append(closest_index)

			# Calculate the widths of the peaks using the data
			peak_widths = [wmin] * len(peak_indices)  # Use the minimum width if peak widths cannot be calculated from the data
			peak_info = {'widths': peak_widths}

		# Calculate the start and end indices of each peak with a larger exclusion area
		start_indices = peak_indices - (exclusion_factor * np.array(peak_info['widths'])).astype(int)
		end_indices = peak_indices + (exclusion_factor * np.array(peak_info['widths'])).astype(int)
		
		# Ensure indices are within data bounds
		start_indices = np.maximum(start_indices, 0)
		end_indices = np.minimum(end_indices, len(x_data) - 1)
		
		# Define the indices covered by the peaks
		covered_indices = []
		for start, end in zip(start_indices, end_indices):
			covered_indices.extend(range(start, end + 1))

		# Remove these indices from the data
		mask = np.ones(x_data.shape[0], dtype = bool)
		mask[covered_indices] = False
	else:
		# if a mask was passed, use that
		mask = fitmask
		peak_indices = None

	uncovered_x_data = x_data[mask]
	uncovered_y_data = y_data[mask]

	# Fit polynomial to the remaining data
	coeff, covar = polynomial_fit(polyorder, uncovered_x_data, uncovered_y_data)

	# Calculate the fitted polynomial values
	bg_values = np.polyval(coeff, x_data)

	# Line subtracted data
	y_data_nobg = y_data - bg_values

	if toplot == True:
		# Plot the data and peaks
		pl.plot(x_data, y_data, label = ' ')

		# Highlight the peaks
		if fitmask is None:
			pl.scatter(x_data[peak_indices], y_data[peak_indices], color = 'green', label = 'peaks')
		else:
			pass

		# Plot the fitted polynomial
		pl.plot(x_data, bg_values, color = 'k', ls = "dashed", label = 'fitted polynomial')

		# Highlight the background used for fitting
		pl.scatter(uncovered_x_data, uncovered_y_data, color = 'red', marker= 'o', alpha = 1, label = 'background used for fit') # type: ignore

		pl.xlabel('sample bias (V)')
		pl.ylabel(' ')
		pl.title('Data plot with peaks, fitted line and background highlighted.')
		pl.legend()
	
	params_used_at_run = {'polyorder': polyorder, 'hmin': hmin, 'hmax': hmax, 'wmin': wmin, 'wmax': wmax, 'prom':prom, 'exclusion_factor': exclusion_factor, 'peak_pos': peak_pos}

	return y_data_nobg, bg_values, coeff, params_used_at_run, mask, covar

def peakfit(xrobj, func = gaussian, fitresult = None, stval = None, bounds = None, toplot = False, pos_x = None, pos_y = None, **kwargs):
	"""Fitting a function to peaks in the data contained in ``xrobj``.

	:param xrobj: :py:mod:`xarray` DataArray, of a single spectrum or a map.
	:type xrobj: :py:mod:`xarray`
	:param func: function to be used for fitting, defaults to gaussian
	:type func: function, optional
	:param fitresult: an :py:mod:`xarray` Dataset of a previous fit calculation, with matching dimensions. If this is passed to :func:`peakfit`, the fit calculation in skipped and the passed Dataset is used.
	:type fitresult: :py:mod:`xarray` Dataset, optional
	:param stval: starting values for the fit parameters of ``func``. You are free to specify only some of the values, the rest will be filled by defaults. Defaults are given in the starting values for keyword arguments in ``func``.
	:type stval: dictionary of ``func`` parameters, optional
	:param bounds: bounds for the fit parameters, used by :py:mod:`xarray.curvefit`. Simlar dictionary, like ``stval``, but the values area a list, with lower and upper components. Defaults to None
	:type bounds: dictionary of ``func`` parameters, with tuples containing lower and upper bounds, optional
	:param toplot: plot the fit result, defaults to ``False``
	:type toplot: boolean, optional
	:param pos_x: pos_x parameter of an :py:mod:`xarray` map to be used in conjunction with ``toplot = True``
	:type pos_x: `int` or `float`, optional
	:param pos_y: pos_y parameter of an :py:mod:`xarray` map to be used in conjunction with ``toplot = True``
	:type pos_y: `int` or `float`, optional
	
	:return: fitted parameters of ``func`` and covariances in a Dataset
	:rtype: :py:mod:`xarray` Dataset

	:Example:

	.. code-block:: python

		import rhkpy

		# example coming soon

	.. note::

		- Use ``toplot`` = `True` to tweak the starting values. If ``toplot`` = `True`, in case of a map, if no ``pos_x`` and ``pos_y`` are specified, the middle of the map is used for plotting.
		- Passing a ``bounds`` dictionary to :func:`peakfit` seems to increase the fitting time significantly. This might be an issue with :py:mod:`xarray.DataArray.curvefit`.
		- By passing a previous fit result, using the optional parameter ``fitresult``, we can just plot the fit result at multiple regions of the map.

	.. seealso::

		It is good practice, to crop the data to the vicinity of the peak you want to fit to.

	"""	
	# get the parameters used by the function: func
	# and also get the default values for the keyword arguments
	param_count = func.__code__.co_argcount
	param_names = func.__code__.co_varnames[:param_count]
	defaults = func.__defaults__
	# get the starting index for the keyword arguments
	kwargs_start_index = param_count - len(defaults)
	# make a dictionary with the keyword arguments (parameters) and their default values specified in the function: func
	kwargs_with_defaults = dict(zip(param_names[kwargs_start_index:], defaults))

	# loop over the keys in stval and fill missing values with defaults
	if stval is None:
		stval = kwargs_with_defaults
	else:
		# if only some values are missing, fill in the rest
		for key in kwargs_with_defaults.keys():
			if key not in stval:
				stval[key] = kwargs_with_defaults[key]
	
	# fit
	# the `xrobj` should have a 'bias' coordinate
	# `nan_policy = omit` skips values with NaN. This is passed to scipy.optimize.curve_fit
	# `max_nfev` is passed to leastsq(). It determines the number of function calls, before quitting.
	if fitresult is None:
		fit = xrobj.curvefit('bias', func, p0 = stval, bounds = bounds, errors = 'ignore', kwargs = {'maxfev': 1000})
	else:
		fit = fitresult

	# plot the resulting fit
	if toplot is True:
		#check if the xrobj is a single spectrum or map
		if 'specpos_x' in xrobj.dims:
			# it's a map
			if (pos_x is not None) and (pos_y is not None):
				# get coordinates to plot, or take the middle spectrum
				plotpos_x = pos_x
				plotpos_y = pos_y
			else:
				wmin = np.min(xrobj.specpos_x.data)
				wmax = np.max(xrobj.specpos_x.data)
				hmin = np.min(xrobj.specpos_y.data)
				hmax = np.max(xrobj.specpos_y.data)
				plotpos_x = (wmax - wmin)/2 + wmin
				plotpos_y = (hmax - hmin)/2 + hmin
			
			paramnames = fit['curvefit_coefficients'].sel(specpos_x = plotpos_x, specpos_y = plotpos_y, method = 'nearest').param.values
			funcparams = fit['curvefit_coefficients'].sel(specpos_x = plotpos_x, specpos_y = plotpos_y, method = 'nearest').data
			# plot the data
			xrobj.sel(specpos_x = plotpos_x, specpos_y = plotpos_y, method = 'nearest').plot(color = 'black', marker = '.', lw = 0, label = 'data')			
		else:
			paramnames = fit['curvefit_coefficients'].param.values
			funcparams = fit['curvefit_coefficients'].data
			# plot the data
			xrobj.plot(marker = 'o', ls = '-', color = 'black', lw = 1, label = 'data')

		# print the starting and fitted values of the parameters
		print('Values of starting parameters: \n', stval, '\n')
		print('Values of fitted parameters:\n')
		for name, fitvalues in zip(paramnames, funcparams):
			print(name, ':', f'{fitvalues:.2f}')

		biasmin = min(xrobj.bias.data)
		biasmax = max(xrobj.bias.data)
		bias_resample = np.linspace(biasmin, biasmax, num = int((biasmax - biasmin)*500))
		
		funcvalues = func(bias_resample, *funcparams)
		# set plotting variables based on the datapoints
		# this should be the bias of the peak maximum if the fit was successful
		fitpeakpos = bias_resample[np.argmax(funcvalues)]
		plotarea_x = (biasmax - biasmin)/2
		plotarea_y = [0.8*np.min(xrobj.data), 1.2*np.max(xrobj.data)]

		pl.plot(bias_resample, funcvalues, color = 'red', lw = 3, alpha = 0.5, label = 'fit')
		pl.xlim([fitpeakpos - plotarea_x, fitpeakpos + plotarea_x])
		pl.ylim(plotarea_y)
		pl.legend()
	
	# copy attributes to the fit dataset, update the 'comments'
	fit.attrs = xrobj.attrs.copy()
	# update the comments if it exists
	# if hasattr(fit, 'comments'):
	# 	fit.attrs['comments'] += 'peak fitting, using ' + str(func.__name__) + '\n'
		
	return fit


def polyflatten(xrobj, field_type = 'topography', **kwargs):
	"""Fits a polynomial to the fast scan lines of topography data and subtracts it from the lines.
	
	The keyword argument ``polyorder`` works the same way as in :func:`bgsubtract`.
	Keywords used by :func:`bgsubtract` can be passed.
	
	Still needs testing.

	:param xrobj: :py:mod:`xarray` image variable of an :class:`~rhkpy.rhkpy_loader.rhkdata` object
	:type xrobj: :py:mod:`xarray` Dataset, :class:`rhkdata.image`
	:param field_type: select the DataArray: 'topography', 'current' or 'lia', defaults to 'topography'
	:type field_type: str, optional
	
	:return: New :class:`rhkdata.image` Dataset of :class:`~rhkpy.rhkpy_loader.rhkdata`, with the DataArray specifiec by ``field_type`` flattened.
	:rtype: :py:mod:`xarray` Dataset
	"""	

	# check if the right object was passed
	# the xrobj passed to the function should always be and image
	if field_type not in xrobj.data_vars:
		print('Wrong xarray type. The data needs to be an `image`')
		return
	
	# make a copy of the xrobject
	flatxrobj = copy.deepcopy(xrobj)

	# iterate through the scan directions of the image
	for scand in flatxrobj.scandir:
		# select the scan direction
		datafield = flatxrobj[field_type].sel(scandir = scand.data)
		# iterate through the slow scan direction lines
		for yy in datafield.y:
			# fit the background
			_, bg_values, _, _, _, _ = bgsubtract(datafield.sel(y = yy).x.data, datafield.sel(y = yy).data, exclusion_factor = 1, **kwargs)
			# subtract the background
			datafield.sel(y = yy).data -= bg_values

	return flatxrobj

## plotting and data visualization -------------------------------------------

def genthumbs(folderpath = '', **kwargs):
	"""Generate thumbnails for the sm4 files present in the current folder (usually the folder where the jupyter notebook is present).
	It ``folderpath`` is specified it generates the thumbnails in the path given.
	All other files are ignored. Subfolders are ignored.
	The method uses :func:`~rhkpy.rhkpy_loader.rhkdata.qplot` to make the png images.

	:param folderpath: path to the folder containing the sm4 files, defaults to ''
	:type folderpath: str, optional

	:Example:
		
		.. code-block:: python

			import rhkpy

			# generate thumbnails of the sm4 files in the current working directory
			rhkpy.genthumbs()

			# generate thumbnails for the folder "stm measurements/maps"
			rhkpy.genthumbs(folderpath = './stm measurements/maps/')

	.. note::

		Possible options for ``folderpath`` are:
		
		- relative path: "./" means the current directory. "../" is one directory above the current one.
		- absolute path: Can start with: "c:/users/averagejoe/data"

		If you use backslashes to separate folder names, remember to append "r" to the beginning of the path to escape backslashes. For example: ``folderpath = r"c:\\users\\averagejoe\\data"``.
		Paths can be copied directly from Windows explorer, if you append an "r".
	"""	
	# import some dependencies
	from .rhkpy_loader import _get_filename, rhkdata

	# make sure folderpath is correct, add a trailing \\ if none is present
	if not re.search(r'\\$', folderpath):
		folderpath += '\\'

	# get the sm4 filenames in the folder
	sm4list = glob.glob(folderpath + '*.sm4')
	filenames = []
	for sm4path in sm4list:
		filenames += [_get_filename(sm4path)]
	
	# generate thumbs
	for fname in filenames:
		# load file
		try:
			data = rhkdata(folderpath + fname)
		except Exception as e:
			# handle the exception
			print('A load error occured in file:', fname, '\n\tThe error is:', e)
			continue
		
		# plot the thumbnail
		data_plot = data.qplot()
		data_plot.save(folderpath + fname[:-4] + '.png')

def navigation(*args, **kwargs):
	"""Takes any number of :class:`~rhkpy.rhkpy_loader.rhkdata` arguments: 'map', 'line', 'spec' and plots all of them on a single plot.
	Plotting of the spectroscopy positions can be skipped by setting the optional keyword: ``plot_spec`` to `False`.
	Plotting is done in the order of passing of the arguments. First argument will be plotted first.

	The color map used for plotting topography images can be specified by the ``cmap`` optional keyword argument. Default value is 'bone'.

	Colors for use in the labels can be specified by the optional keyword: ``palette_name``. If this is used, the number of colors also needs to be specified, by: ``num_colors``.
	For possible palette options look at the `bokeh palettes <https://docs.bokeh.org/en/latest/docs/reference/palettes.html>`_.

	:return: :py:mod:`holoviews` plot
	:rtype: :py:mod:`holoviews`

	:Example:
		
		.. code-block:: python

			import rhkpy

			# Load some data
			didvmap = rhkpy.rhkdata('didvmap path/map.sm4')
			topography = rhkpy.rhkdata('topo path/topo1.sm4')
			single_spec = rhkpy.rhkdata('single spec path/single spec.sm4')

			# plot the topography and spectroscopy positions
			rhkpy.navigation(topography, didvmap, single_spec)

			# skip plotting the spectroscopy positions
			rhkpy.navigation(topography, didvmap, single_spec, plot_spec = False)

			# In the above examples, the image from topography
			# is plotted before the image data of didivmap!
			# You can change the plotting order by changing
			# the order of the `rhkdata` instances in the arguments.
			rhkpy.navigation(didvmap, topography, single_spec) # now didvmap is plotted first

	.. note::

		Arguments are plotted in the order they are passed to :func:`navigation`.

	"""	
	# arguments should be rhkdata instances
	# take care of optional keyword arguments
	# an optional argument is plot_spec. If False, the spectroscopy positions are not plotted
	if 'plot_spec' not in kwargs:
		plot_spec = True
	else:
		plot_spec = kwargs['plot_spec']
	if 'cmap' not in kwargs:
		cmap = 'bone'
	else:
		cmap = kwargs['cmap']
	if 'palette_name' not in kwargs:
		palette_name = 'Category10'
	else:
		palette_name = kwargs['palette_name']
	if 'num_colors' not in kwargs:
		num_colors = 10
	else:
		num_colors = kwargs['num_colors']

	# we need a function to plot a bounding box around the topo data
	def bounding_box(rhkdata_obj, c):
		l_top = rhkdata_obj.image.topography.drop('scandir')[-1, :].hvplot.line(x = 'x', y = 'y', color = c, label = rhkdata_obj.image.attrs['filename'])
		l_bottom = rhkdata_obj.image.topography.drop('scandir')[0, :].hvplot.line(x = 'x', y = 'y', color = c)
		l_left = rhkdata_obj.image.topography.drop('scandir')[:, 0].hvplot.line(x = 'x', y = 'y', color = c)
		l_right = rhkdata_obj.image.topography.drop('scandir')[:, -1].hvplot.line(x = 'x', y = 'y', color = c)
		return l_bottom * l_left * l_right * l_top

	def plot_spec_positions(rhkdata_obj, c):
		if rhkdata_obj.datatype == 'map':
			if rhkdata_obj.spectype == 'iv':
				_ = rhkdata_obj.spectra.drop(['bias', 'repetitions', 'biasscandir']).drop_vars(['lia', 'current'])
				specplot = _.hvplot.scatter(x = 'x', y = 'y', groupby = [], marker = 'x', color = c, label = 'spec pos: ' + rhkdata_obj.image.attrs['filename'])
			elif rhkdata_obj.spectype == 'iz':
				_ = rhkdata_obj.spectra.drop(['z', 'repetitions', 'zscandir']).drop_vars(['current'])
				specplot = _.hvplot.scatter(x = 'x', y = 'y', groupby = [], marker = 'x', color = c, label = 'spec pos: ' + rhkdata_obj.image.attrs['filename'])
		elif rhkdata_obj.datatype == 'line':
			if rhkdata_obj.spectype == 'iv':
				_ = rhkdata_obj.spectra.drop_vars(['lia', 'current']).drop(['biasscandir', 'repetitions', 'bias'])
				specplot = _.hvplot.scatter(x = 'x', y = 'y', groupby = [], marker = 'x', color = c, label = 'spec pos: ' + rhkdata_obj.spectra.attrs['filename'])
			elif rhkdata_obj.spectype == 'iz':
				_ = rhkdata_obj.spectra.drop_vars(['current']).drop(['zscandir', 'repetitions', 'z'])
				specplot = _.hvplot.scatter(x = 'x', y = 'y', groupby = [], marker = 'x', color = c, label = 'spec pos: ' + rhkdata_obj.spectra.attrs['filename'])
		elif rhkdata_obj.datatype == 'spec':
			specplot = rhkdata_obj.spectra.hvplot.scatter(x = 'x', y = 'y', groupby = [], marker = 'x', color = c, size = 200, line_width = 3, label = 'spec pos: ' + rhkdata_obj.spectra.attrs['filename'])
		else:
			# it should never get to this point
			specplot = hv.Empty()
		return specplot

	if len(args) == 0:
		print('Please supply some rhkdata instances to plot.')
		return

	datatypes = []
	spectypes = []
	for stmdata in args:
		datatypes += [stmdata.datatype]
		spectypes += [stmdata.spectype]

	# get the indices, where the datatype has an image or has spectroscopy data
	indices_topo = []
	indices_spec = []
	for index, value in enumerate(datatypes):
		if value in ('map', 'image'):
			indices_topo += [index]
		if value in ('map', 'line', 'spec'):
			indices_spec += [index]
	# indices_topo = [index for index, value in enumerate(datatypes) if value in ('map', 'image')]
	
	# plot those images
	# get a color map for the bounding boxes
	from bokeh.palettes import all_palettes
	from itertools import cycle
	colors = all_palettes[palette_name][num_colors]
	# make the color list cyclic
	color_cycle = cycle(colors)

	# do the first plot
	if indices_topo != []:
		# plot the first one
		topo_abs = args[indices_topo[0]].coord_to_absolute()
		navi_plot = topo_abs._qplot_topo(cmap_topo = cmap)
		# draw a bounding box
		navi_plot *= bounding_box(topo_abs, next(color_cycle))
		for i in indices_topo[1:]:
			topo_abs = args[i].coord_to_absolute()
			navi_plot *= topo_abs._qplot_topo(cmap_topo = cmap, clabel = 'height (nm)')
			# plot a bounding box around the image
			navi_plot *= bounding_box(topo_abs, next(color_cycle))
		
	# if plot_spec is True, plot the spectra positions
	if plot_spec is True:
		if indices_spec != []:
			# plot the first one if no topography data was passed
			if indices_topo == []:
				navi_plot = plot_spec_positions(args[indices_spec[0]], next(color_cycle))
			else:
				navi_plot *= plot_spec_positions(args[indices_spec[0]], next(color_cycle))
			# for datatypes containing a spectrum, plot the spectrum positions
			for i in indices_spec[1:]:
				navi_plot *= plot_spec_positions(args[i], next(color_cycle))

	return navi_plot.opts(frame_width = 400, frame_height = 400, legend_position = 'right', title = ' ')
