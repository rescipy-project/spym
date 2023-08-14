import matplotlib.pyplot as pl
import numpy as np
import xarray as xr
import hvplot
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import ndimage

def conf_hvplot_defaults():
	"""Set up some default values for the plotting parameters, when using `hvplot`.
	"""	
	hvplot.extension('bokeh')
	# Setting some default plot options for hvplot
	from holoviews import opts
	opts.defaults(
		opts.Image(
			aspect = 1
			)
		) # can't set default colormap with this

def coord_to_absolute(xrobj):
	# the xrobj passed to the function should always be and image
	# if not hasattr(xrobj, 'image'):
	# 	print('Wrong xarray type. The data needs to be an `image`, not `spectra`')
	# 	return
	
	# TODO here we will need to loop trough the dataarrays in xrobj
	# for now let's just take the forward topo
	imagedata = xrobj.data

	rotated = ndimage.rotate(
		imagedata,
		30,
		reshape = True,
		mode = 'constant',
		cval = 0
		)
	
	# relcoord_xrobj = xrobj.copy()
	# relcoord_xrobj.data = rotated

	return rotated

def plot_spec_position(stmdata_object, repetitions, zscandir, z, **kwargs):
	
	# specpos_plot = stmdata_object.spectra.isel(repetitions=repetitions, zscandir=zscandir).sel(z = z, method='nearest').hvplot(data_aspect=1, cmap='magma')
	# return specpos_plot

	#### Using Matplotlib
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
	# pl.scatter(stmdata_object.spectra['x'].data, stmdata_object.spectra['y'].data,
	pl.scatter(stmdata_object.spectra['x'].data, stmdata_object.spectra['y'].data,
		marker = '.',
		alpha = 0.5,
		color = 'r'
		)
	
## peak finding and background subtraction

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

def bgsubtract(x_data, y_data, polyorder = 3, toplot = False, fitmask = None, hmin = 0.5, hmax = 10000, wmin = 1.5, wmax = 20, prom = 2, exclusion_factor = 3, peak_pos = None):
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



