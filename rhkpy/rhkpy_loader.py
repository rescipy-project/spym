import matplotlib.pyplot as pl
import numpy as np
import xarray as xr
import re

# import load from spym
from spym.io import load

## Using the old loader
from spym.io import rhksm4

## flatten and plane fitting
from spym.process.level import align
from spym.process.level import plane

from .rhkpy_process import *

class rhkdata:
	"""
	A container for the xarray based structure of the RHK data. Loads the RHK "sm4" file from the path at: ``filename``.

	:param filename: path and filename of the "sm4" file to be loaded
	:type filename: str
	:param repetitions: The number of repeated aquisitions of spectra in tip position, defaults to 0
	:type repetitions: int, optional
	:param alternate: `True` if the bias is swept forward and backward, `False` if not, defaults to True
	:type alternate: bool, optional
	:param datatype: datatype, defaults to None
	:type datatype: str, optional
	:param spectype: spectrum type, defaults to None
	:type spectype: str, optional

	Some variables of the :class:`rhkdata` class:

	:var filename: (type str) filename of the "sm4" file
	:var image: (type :py:mod:`xarray` Dataset) Dataset containing the image data
	:var spectra: (type :py:mod:`xarray` Dataset) Dataset containing the spectroscopy data
	:var spymdata: (type :py:mod:`spym` instance) Dataset, as loaded by the :py:mod:`spym` module

	All the variables can be listed by calling: :class:`rhkdata.print_info`.

	.. note::
		Optional parameters are not needed, they are just present for debugging purposes. Their values are inferred from the sm4 file.
	
	:Example:
		
		.. code-block:: python

			import rhkpy

			# example coming soon

	"""
	def __init__(self, filename, repetitions = 0, alternate = True, datatype = None, spectype = None, **kwargs):
		"""Initialize the :class:`rhkdata` instance
		"""		

		if isinstance(alternate, bool) == False:
			print("alternate needs to be a bool variable: True or False. Default is True")

		self.filename = filename

		# Boolean value, True if alternate scan directions is turned on
		self.alternate = alternate

		# Load the data using spym
		self.spymdata = load_spym(self.filename)

		# check software version. Not tested for MinorVer < 6
		l = list(self.spymdata.keys())
		if self.spymdata[l[-1]].attrs['RHK_MinorVer'] < 6:
			print('stmdatastruct not tested for RHK Rev version < 6. Some things might not work as expected.')

		# check type of data and spectra contained in the file, if no type is specified
		if datatype is None:
			self.datatype, self.spectype = _checkdatatype(self)
		else:
			if (datatype != 'map') and (datatype != 'line') and (datatype != 'spec') and (datatype != 'image'):
				print('datatype must be either: map, line, spec or image')
				return
			else:
				self.datatype = datatype
				self.spectype = spectype

		# number of spectra at a tip position
		# default value is 0, if this is changed, the code will use the given value, othewise it will try to infer the number of repetitions from the number of identical tip positions
		if self.datatype != 'image':
			if repetitions != 0:
				# overwrite the default value and the value inferred from tip coordinates
				# check if parameters passed to the class are valid
				if repetitions <= 0:
					print("repetitions needs to be an integer, with a value of 1 or above. Default is 1")
				elif isinstance(repetitions, int) == False:
					print("repetitions needs to be an integer. Default is 1")
				self.repetitions = repetitions
			else:
				# determine the number of repetitions from the number of indentical tip coordinates in the beginning of RHK_SpecDrift_Xcoord
				self.repetitions = _checkrepetitions(self)
		else:
			self.repetitions = repetitions

		# load data into xarray, for all data types
		if self.datatype == 'map':
			self = _load_specmap(self)
		elif self.datatype == 'line':
			self = _load_line(self)
		elif self.datatype == 'spec':
			self = _load_spec(self)
		elif self.datatype == 'image':
			self = _load_image(self)

	def print_info(self):
		"""List the variables of the :class:`rhkdata` instance.
		"""
		for item in self.__dict__:
			print(item)
		if 'image' in self.__dict__:
			print('\nimage:')
			for item in self.image.data_vars:
				print('\t', item)
		if 'spectra' in self.__dict__:
			print('\nspectra:')
			for item in self.spectra.data_vars:
				print('\t', item)
		print('\nspymdata:')
		for item in self.spymdata:
			print('\t', item)

	def plot_specpos(self):		
		# plot the positions of spectra on a topography image
		return plot_specpos(self)

### internal functions -----------------------------------------------------------

def _checkrepetitions(stmdata_object):
	coordlist = stmdata_object.spymdata.Current.attrs['RHK_SpecDrift_Xcoord']
	reps = 0
	for coo in coordlist:
		if coo == coordlist[0]:
			reps += 1
		else:
			break
	reps = int(reps / (stmdata_object.alternate + 1))
	return reps

def _checkdatatype(stmdata_object):
	# check if the list of keys in the spym data contain 'Current'
	# If yes, it is not a pure topo image
	l = list(stmdata_object.spymdata.keys())
	if 'Current' in l:
		# it's a single spec, line spec or map
		if stmdata_object.spymdata['Current'].attrs['RHK_PageType'] == 38:
			stmdata_object.datatype = 'spec'
			# determine if it's Iz or dI/dV
			if stmdata_object.spymdata['Current'].attrs['RHK_LineType'] == 7:
				stmdata_object.spectype = 'iv'
			elif stmdata_object.spymdata['Current'].attrs['RHK_LineType'] == 8:
				stmdata_object.spectype = 'iz'
		elif stmdata_object.spymdata['Current'].attrs['RHK_PageType'] == 16:
			# this can be either a line spectrum or a map
			# decide based on the aspect ratio of the spectroscopy tip positions
			xcoo = np.array(stmdata_object.spymdata['Current'].attrs['RHK_SpecDrift_Xcoord'])
			ycoo = np.array(stmdata_object.spymdata['Current'].attrs['RHK_SpecDrift_Ycoord'])
			if _aspect_ratio(xcoo, ycoo) > 10:
				stmdata_object.datatype = 'line'
			else:
				stmdata_object.datatype = 'map'
			# determine if it's Iz or dI/dV
			if stmdata_object.spymdata['Current'].attrs['RHK_LineType'] == 7:
				stmdata_object.spectype = 'iv'
			elif stmdata_object.spymdata['Current'].attrs['RHK_LineType'] == 8:
				stmdata_object.spectype = 'iz'
	else:
		stmdata_object.datatype = 'image'
		stmdata_object.spectype = None
	
	return stmdata_object.datatype, stmdata_object.spectype


def _aspect_ratio(x, y):
    xy = np.stack((x, y), axis=0)
    eigvals, eigvecs = np.linalg.eig(np.cov(xy))
    center = xy.mean(axis=-1)
    for val, vec in zip(eigvals, eigvecs.T):
        val *= 2
        xcov,ycov = np.vstack((center + val * vec, center, center - val * vec)).T
    aspect = max(eigvals) / min(eigvals)
    return aspect


def _get_filename(s):
	# If the string ends with a slash or backslash, remove it first
    s = s.rstrip("/\\")
    # Then, match any character other than backslash or slash until the end of the string
    match = re.search(r'[^/\\]+$', s)
    return match.group(0) if match else None


def _load_specmap(stmdata_object):
	# total number of spectra in one postion of the tip
	stmdata_object.numberofspectra = int((stmdata_object.alternate + 1)*stmdata_object.repetitions)
	# load the image
	stmdata_object = _load_image(stmdata_object)

	# decide if it's a dI/dV or I(z) map
	if stmdata_object.spectype == 'iv':
		# create a DataSet, containing the LIA and Current maps, with appropriate position coordinates
		stmdata_object = _xr_map_iv(stmdata_object)
		# add metadata to the xarray
		stmdata_object = _add_map_metadata(stmdata_object)
	elif stmdata_object.spectype == 'iz':
		# create xarray Dataset
		stmdata_object = _xr_map_iz(stmdata_object)
		# add metadata to the xarray
		stmdata_object = _add_map_metadata(stmdata_object)
	return stmdata_object


def _load_line(stmdata_object):
	# total number of spectra in one postion of the tip
	stmdata_object.numberofspectra = int((stmdata_object.alternate + 1)*stmdata_object.repetitions)
	# load the image data
	stmdata_object = _load_image(stmdata_object)

	# decide if it's a dI/dV or I(z) line
	if stmdata_object.spectype == 'iv':
		stmdata_object = _xr_line_iv(stmdata_object)
		stmdata_object = _add_line_metadata(stmdata_object)
	elif stmdata_object.spectype == 'iz':
		stmdata_object = _xr_line_iz(stmdata_object)
		stmdata_object = _add_spec_metadata(stmdata_object)
	return stmdata_object


def _load_spec(stmdata_object):
	# in this case the total number of spectra can be inferred
	# total number of spectra in one postion of the tip
	stmdata_object.repetitions = int(stmdata_object.spymdata.Current.data.shape[1] / (stmdata_object.alternate + 1))
	stmdata_object.numberofspectra = int((stmdata_object.alternate + 1)*stmdata_object.repetitions)

	# decide if it's a dI/dV or I(z) spec
	if stmdata_object.spectype == 'iv':
		stmdata_object = _xr_spec_iv(stmdata_object)
		stmdata_object = _add_spec_metadata(stmdata_object)
	elif stmdata_object.spectype == 'iz':
		stmdata_object = _xr_spec_iz(stmdata_object)
		stmdata_object = _add_spec_metadata(stmdata_object)
	return stmdata_object


def _load_image(stmdata_object):
	# load the image data
	stmdata_object = _xr_image(stmdata_object)
	# add metadata
	stmdata_object = _add_image_metadata(stmdata_object)
	return stmdata_object


def _xr_map_iv(stmdata_object):
	"""
	TODO need to change spectrum rearranging for the case where alternate is False

	Create a DataSet containing the Lock-In (LIA) and Current spectroscopy data
	Use the absolute values of the tip positions as coordinates

	In spym the spectroscopy data is loaded into an array,
	which has axis=0 the number of datapoints in the spectra
	and axis=1 the number of spectra in total.

	When rearranging, the number of repetitions within each tip position is assumed to be 1
	and alternate scan direction is assumed to be turned on.
	These options can be changed by the parameters, `repetitions` and `alternate`
	"""

	# extract the numpy array containing the LIA data from the spym object
	specarray = stmdata_object.spymdata.LIA_Current.data
	# extract the numpy array containing the Current data from the spym object
	currentarray = stmdata_object.spymdata.Current.data

	# total number of spectra in one postion of the tip
	numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions
	# size of the map in mapsize x mapsize
	mapsize = int(np.sqrt(specarray.shape[1] / numberofspectra))

	# reshape LIA data
	# collect all spectra measured in the same `X, Y` coordinate into an axis (last) of an array.
	temp = np.reshape(specarray, (specarray.shape[0], -1, numberofspectra), order='C')
	# Every other spectrum is a forward and backward scan in bias sweep. Separate the forward and backward scans into differing arrays by slicing.
	# These are all the forward and backward bias sweep spectra, arranged along axis=1, with axis=2 being the repetitions
	spec_fw = temp[:, :, 0::2]
	spec_bw = temp[:, :, 1::2]
	# reshape the forward and backward parts into a map
	speccmap_fw = np.reshape(spec_fw, (spec_fw.shape[0], mapsize, mapsize, spec_fw.shape[2]), order='C')
	speccmap_bw = np.reshape(spec_bw, (spec_bw.shape[0], mapsize, mapsize, spec_bw.shape[2]), order='C')
	
	# The last axis (in this case with length of 1) contains the repeated scans in one particular pixel.
	# If the `repetitions` variable is set to greater than 1, this will contains the repeated spectra within an `X, Y` pixel.
	# The array needs to be flipped along axis = 1 (the "x" axis in the topography image) to fit with the data read by the ASCII method
	
	liafw = np.flip(speccmap_fw, axis=1)
	liabw = np.flip(speccmap_bw, axis=1)

	# reshape Current data
	temp = np.reshape(currentarray, (currentarray.shape[0], -1, numberofspectra), order='C')
	# Every other spectrum is a forward and backward scan in bias sweep. Separate the forward and backward scans into differing arrays by slicing.
	# These are all the forward and backward bias sweep spectra, arranged along axis=1, with axis=2 being the repetitions
	current_fw = temp[:, :, 0::2]
	current_bw = temp[:, :, 1::2]
	# reshape the forward and backward parts into a map
	currentmap_fw = np.reshape(current_fw, (current_fw.shape[0], mapsize, mapsize, current_fw.shape[2]), order='C')
	currentmap_bw = np.reshape(current_bw, (current_bw.shape[0], mapsize, mapsize, current_bw.shape[2]), order='C')
	
	# The last axis (in this case with length of 1) contains the repeated scans in one particular pixel.
	# If the `repetitions` variable is set to greater than 1, this will contains the repeated spectra within an `X, Y` pixel.
	# The array needs to be flipped along axis = 1 (the "x" axis in the topography image) to fit with the data read by the ASCII method
	
	currentfw = np.flip(currentmap_fw, axis=1)
	currentbw = np.flip(currentmap_bw, axis=1)	

	# Coordinates of the spectroscopy map
	
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	xcoo = np.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = np.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Ycoord'])
	
	# reshaping the coordinates similarly to the spectra. This is a coordinates mesh
	# at the end slicing the arrays to get the X, Y coordinates, we don't need the mesh
	meshx = np.reshape(xcoo, (mapsize, mapsize, numberofspectra), order='C')[:, :, 0]
	meshy = np.reshape(ycoo, (mapsize, mapsize, numberofspectra), order='C')[:, :, 0]
	tempx = np.reshape(xcoo, (mapsize, mapsize, numberofspectra), order='C')[0, :, 0]
	tempy = np.reshape(ycoo, (mapsize, mapsize, numberofspectra), order='C')[:, 0, 0]

	# Constructing the xarray DataSet 
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding specific attributes
	xrspec = xr.Dataset(
		data_vars = dict(
			lia = (['bias', 'specpos_x', 'specpos_y', 'repetitions', 'biasscandir'], np.stack((liafw, liabw), axis=-1)*10**12),
			current = (['bias', 'specpos_x', 'specpos_y', 'repetitions', 'biasscandir'], np.stack((currentfw, currentbw), axis=-1)*10**12),
			x = (['specpos_x', 'specpos_y'], meshx*10**9),
			y = (['specpos_x', 'specpos_y'], meshy*10**9)
			),
		coords = dict(
			bias = stmdata_object.spymdata.coords['LIA_Current_x'].data,
			specpos_x = tempx*10**9,
			specpos_y = tempy*10**9,
			repetitions = np.array(range(stmdata_object.repetitions)),
			biasscandir = np.array(['left', 'right'], dtype = 'U')
			),
		attrs = dict(filename = _get_filename(stmdata_object.filename))
	)

	xrspec['lia'].attrs['units'] = 'pA'
	xrspec['lia'].attrs['long units'] = 'picoampere'
	xrspec['current'].attrs['units'] = 'pA'
	xrspec['current'].attrs['long units'] = 'picoampere'
	xrspec['x'].attrs['units'] = 'nm'
	xrspec['x'].attrs['long units'] = 'nanometer'
	xrspec['y'].attrs['units'] = 'nm'
	xrspec['y'].attrs['long units'] = 'nanometer'
	xrspec.coords['bias'].attrs['units'] = 'V'
	xrspec.coords['specpos_x'].attrs['units'] = 'nm'
	xrspec.coords['specpos_y'].attrs['units'] = 'nm'
	xrspec.coords['specpos_x'].attrs['long units'] = 'nanometer'
	xrspec.coords['specpos_y'].attrs['long units'] = 'nanometer'

	stmdata_object.spectra = xrspec
	return stmdata_object


def _xr_line_iv(stmdata_object):
	"""
	Create a DataSet containing the Lock-In (LIA) and Current spectroscopy data
	Use the absolute values of the tip positions as coordinates

	In spym the spectroscopy data is loaded into an array,
	which has axis=0 the number of datapoints in the spectra
	and axis=1 the number of spectra in total.

	When rearranging, the number of repetitions within each tip position is assumed to be 1
	and alternate scan direction is assumed to be turned on.
	These options can be changed by the parameters, `repetitions` and `alternate`
	"""

	# extract the numpy array containing the LIA data from the spym object
	specarray = stmdata_object.spymdata.LIA_Current.data
	# extract the numpy array containing the Current data from the spym object
	currentarray = stmdata_object.spymdata.Current.data

	# total number of spectra in one postion of the tip
	numberofspectra = int((stmdata_object.alternate + 1)*stmdata_object.repetitions)
	# size of the line, the number of the different physical positions of the tip
	linesize = int(specarray.shape[1] / numberofspectra)

	# reshape LIA data
	# Every other spectrum is a forward and backward scan in bias sweep. Separate the forward and backward scans into differing arrays by slicing.
	# These are all the forward and backward bias sweep spectra, arranged along axis=1, with axis=2 being the repetitions
	templia = np.reshape(specarray, (specarray.shape[0], -1, numberofspectra), order='C')
	liafw = templia[:, :, 0::2]
	liabw = templia[:, :, 1::2]

	# reshape Current data
	tempcurr = np.reshape(currentarray, (currentarray.shape[0], -1, numberofspectra), order='C')
	currentfw = tempcurr[:, :, 0::2]
	currentbw = tempcurr[:, :, 1::2]

	"""
	Coordinates of the spectroscopy map
	"""
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	xcoo = np.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = np.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra. Need only every nth coordinate, where n is then number of spectra in a tip position
	tempx = xcoo[0::numberofspectra]
	tempy = ycoo[0::numberofspectra]
	linelength = np.sqrt((tempx[-1] - tempx[0])**2 + (tempy[-1] - tempy[0])**2)
	# distance coordinates along the line in [nm]
	linecoord = np.linspace(0, linelength, num=tempx.shape[0])*10**9

	"""
	Constructing the xarray Dataset 
	"""
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding specific attributes
	xrspec = xr.Dataset(
		data_vars = dict(
			lia = (['bias', 'dist', 'repetitions', 'biasscandir'], np.stack((liafw, liabw), axis=-1)*10**12),
			current = (['bias', 'dist', 'repetitions', 'biasscandir'], np.stack((currentfw, currentbw), axis=-1)*10**12),
			x = (['dist'], tempx*10**9),
			y = (['dist'], tempy*10**9)
			),
		coords = dict(
			bias = stmdata_object.spymdata.coords['LIA_Current_x'].data,
			dist = linecoord,
			repetitions = np.array(range(stmdata_object.repetitions)),
			biasscandir = np.array(['left', 'right'], dtype = 'U')
			),
		attrs = dict(filename = _get_filename(stmdata_object.filename))
	)

	xrspec.coords['dist'].attrs['units'] = 'nm'
	xrspec.coords['dist'].attrs['long units'] = 'nanometer'

	xrspec['x'].attrs['units'] = 'nm'
	xrspec['y'].attrs['units'] = 'nm'
	xrspec['x'].attrs['long units'] = 'nanometer'
	xrspec['y'].attrs['long units'] = 'nanometer'
	xrspec['lia'].attrs['units'] = 'pA'
	xrspec['lia'].attrs['long units'] = 'picoampere'
	xrspec['current'].attrs['units'] = 'pA'
	xrspec['current'].attrs['long units'] = 'picoampere'

	stmdata_object.spectra = xrspec
	return stmdata_object


def _xr_spec_iv(stmdata_object):
	"""
	Create a DataSet containing the Lock-In (LIA) and Current spectroscopy data
	Use the absolute values of the tip positions are in the attributes
	"""

	# extract the numpy array containing the LIA data from the spym object
	specarray = stmdata_object.spymdata.LIA_Current.data
	# extract the numpy array containing the Current data from the spym object
	currentarray = stmdata_object.spymdata.Current.data

	# reshape LIA data
	# Every other spectrum is a forward and backward scan in bias sweep. Separate the forward and backward scans into differing arrays by slicing.
	liafw = specarray[:, 0::2]
	liabw = specarray[:, 1::2]

	# reshape Current data
	currentfw = currentarray[:, 0::2]
	currentbw = currentarray[:, 1::2]

	# Coordinates of the spectroscopy map
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	# Here we only need the first x and y components
	xcoo = np.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = np.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra.
	tempx = xcoo[0]
	tempy = ycoo[0]

	# Constructing the xarray Dataset 
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding specific attributes
	xrspec = xr.Dataset(
		data_vars = dict(
			lia = (['bias', 'repetitions', 'biasscandir'], np.stack((liafw, liabw), axis=-1)*10**12),
			current = (['bias', 'repetitions', 'biasscandir'], np.stack((currentfw, currentbw), axis=-1)*10**12),
			x = tempx*10**9,
			y = tempy*10**9
			),
		coords = dict(
			bias = stmdata_object.spymdata.coords['LIA_Current_x'].data,
			repetitions = np.array(range(stmdata_object.repetitions)),
			biasscandir = np.array(['left', 'right'], dtype = 'U')
			),
		attrs = dict(filename = _get_filename(stmdata_object.filename))
	)

	xrspec.attrs['speccoord_x'] = tempx*10**9
	xrspec.attrs['speccoord_y'] = tempy*10**9
	xrspec.attrs['speccoord_x units'] = 'nm'
	xrspec.attrs['speccoord_y units'] = 'nm'

	xrspec['lia'].attrs['units'] = 'pA'
	xrspec['lia'].attrs['long units'] = 'picoampere'
	xrspec['current'].attrs['units'] = 'pA'
	xrspec['current'].attrs['long units'] = 'picoampere'

	stmdata_object.spectra = xrspec
	return stmdata_object


def _xr_map_iz(stmdata_object):
	"""
	TODO need to change spectrum rearranging for the case where alternate is False

	Create a DataSet containing the Lock-In (LIA) and Current spectroscopy data
	Use the absolute values of the tip positions as coordinates

	In spym the spectroscopy data is loaded into an array,
	which has axis=0 the number of datapoints in the spectra
	and axis=1 the number of spectra in total.

	When rearranging, the number of repetitions within each tip position is assumed to be 1
	and alternate scan direction is assumed to be turned on.
	These options can be changed by the parameters, `repetitions` and `alternate`
	"""

	# extract the numpy array containing the Current data from the spym object
	currentarray = stmdata_object.spymdata.Current.data

	# total number of spectra in one postion of the tip
	numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions
	# size of the map in mapsize x mapsize
	mapsize = int(np.sqrt(currentarray.shape[1] / numberofspectra))

	# reshape Current data
	temp = np.reshape(currentarray, (currentarray.shape[0], -1, numberofspectra), order='C')
	# Every other spectrum is a forward and backward scan in bias sweep. Separate the forward and backward scans into differing arrays by slicing.
	# These are all the forward and backward bias sweep spectra, arranged along axis=1, with axis=2 being the repetitions
	current_fw = temp[:, :, 0::2]
	current_bw = temp[:, :, 1::2]
	# reshape the forward and backward parts into a map
	currentmap_fw = np.reshape(current_fw, (current_fw.shape[0], mapsize, mapsize, current_fw.shape[2]), order='C')
	currentmap_bw = np.reshape(current_bw, (current_bw.shape[0], mapsize, mapsize, current_bw.shape[2]), order='C')
	"""
	The last axis (in this case with length of 1) contains the repeated scans in one particular pixel.
	If the `repetitions` variable is set to greater than 1, this will contains the repeated spectra within an `X, Y` pixel.
	The array needs to be flipped along axis = 1 (the "x" axis in the topography image) to fit with the data read by the ASCII method
	"""
	currentfw = np.flip(currentmap_fw, axis=1)
	currentbw = np.flip(currentmap_bw, axis=1)	

	"""
	Coordinates of the spectroscopy map
	"""
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	xcoo = np.array(stmdata_object.spymdata.Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = np.array(stmdata_object.spymdata.Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra. This is a coordinates mesh
	# at the end slicing the arrays to get the X, Y coordinates, we don't need the mesh
	meshx = np.reshape(xcoo, (mapsize, mapsize, numberofspectra), order='C')[:, :, 0]
	meshy = np.reshape(ycoo, (mapsize, mapsize, numberofspectra), order='C')[:, :, 0]
	tempx = np.reshape(xcoo, (mapsize, mapsize, numberofspectra), order='C')[0, :, 0]
	tempy = np.reshape(ycoo, (mapsize, mapsize, numberofspectra), order='C')[:, 0, 0]

	"""
	Constructing the xarray DataSet 
	"""
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding specific attributes
	xrspec = xr.Dataset(
		data_vars = dict(
			current = (['z', 'specpos_x', 'specpos_y', 'repetitions', 'zscandir'], np.stack((currentfw, currentbw), axis=-1)*10**12),
			x = (['specpos_x', 'specpos_y'], meshx*10**9),
			y = (['specpos_x', 'specpos_y'], meshy*10**9)
			),
		coords = dict(
			z = stmdata_object.spymdata.coords['Current_x'].data*10**9,
			specpos_x = tempx*10**9,
			specpos_y = tempy*10**9,
			repetitions = np.array(range(stmdata_object.repetitions)),
			zscandir = np.array(['up', 'down'], dtype = 'U')
			),
		attrs = dict(filename = _get_filename(stmdata_object.filename))
	)

	xrspec['current'].attrs['units'] = 'pA'
	xrspec['current'].attrs['long units'] = 'picoampere'
	xrspec['x'].attrs['units'] = 'nm'
	xrspec['x'].attrs['long units'] = 'nanometer'
	xrspec['y'].attrs['units'] = 'nm'
	xrspec['y'].attrs['long units'] = 'nanometer'
	xrspec.coords['z'].attrs['units'] = 'nm'
	xrspec.coords['z'].attrs['long units'] = 'nanometer'
	xrspec.coords['specpos_x'].attrs['units'] = 'nm'
	xrspec.coords['specpos_y'].attrs['units'] = 'nm'
	xrspec.coords['specpos_x'].attrs['long units'] = 'nanometer'
	xrspec.coords['specpos_y'].attrs['long units'] = 'nanometer'

	stmdata_object.spectra = xrspec
	return stmdata_object


def _xr_line_iz(stmdata_object):
	"""
	Create a DataSet containing the Lock-In (LIA) and Current spectroscopy data
	Use the absolute values of the tip positions as coordinates

	In spym the spectroscopy data is loaded into an array,
	which has axis=0 the number of datapoints in the spectra
	and axis=1 the number of spectra in total.

	When rearranging, the number of repetitions within each tip position is assumed to be 1
	and alternate scan direction is assumed to be turned on.
	These options can be changed by the parameters, `repetitions` and `alternate`
	"""

	# extract the numpy array containing the Current data from the spym object
	currentarray = stmdata_object.spymdata.Current.data

	# total number of spectra in one postion of the tip
	numberofspectra = int((stmdata_object.alternate + 1)*stmdata_object.repetitions)
	# size of the line, the number of the different physical positions of the tip
	linesize = int(currentarray.shape[1] / numberofspectra)

	# reshape Current data
	tempcurr = np.reshape(currentarray, (currentarray.shape[0], -1, numberofspectra), order='C')
	currentfw = tempcurr[:, :, 0::2]
	currentbw = tempcurr[:, :, 1::2]

	"""
	Coordinates of the spectroscopy map
	"""
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	xcoo = np.array(stmdata_object.spymdata.Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = np.array(stmdata_object.spymdata.Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra. Need only every nth coordinate, where n is the number of spectra in a position
	tempx = xcoo[0::numberofspectra]
	tempy = ycoo[0::numberofspectra]
	linelength = np.sqrt((tempx[-1] - tempx[0])**2 + (tempy[-1] - tempy[0])**2)
	# distance coordinates along the line in [nm]
	linecoord = np.linspace(0, linelength, num=tempx.shape[0])*10**9

	"""
	Constructing the xarray Dataset 
	"""
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding specific attributes
	xrspec = xr.Dataset(
		data_vars = dict(
			current = (['z', 'dist', 'repetitions', 'zscandir'], np.stack((currentfw, currentbw), axis=-1)*10**12),
			x = (['dist'], tempx*10**9),
			y = (['dist'], tempy*10**9)
			),
		coords = dict(
			z = stmdata_object.spymdata.coords['Current_x'].data*10**9,
			dist = linecoord,
			repetitions = np.array(range(stmdata_object.repetitions)),
			zscandir = np.array(['up', 'down'], dtype = 'U')
			),
		attrs = dict(filename = _get_filename(stmdata_object.filename))
	)

	xrspec.coords['dist'].attrs['units'] = 'nm'
	xrspec.coords['dist'].attrs['long units'] = 'nanometer'
	xrspec.coords['z'].attrs['units'] = 'nm'
	xrspec.coords['z'].attrs['long units'] = 'nanometer'

	xrspec['x'].attrs['units'] = 'nm'
	xrspec['y'].attrs['units'] = 'nm'
	xrspec['x'].attrs['long units'] = 'nanometer'
	xrspec['y'].attrs['long units'] = 'nanometer'
	xrspec['current'].attrs['units'] = 'pA'
	xrspec['current'].attrs['long units'] = 'picoampere'

	stmdata_object.spectra = xrspec
	return stmdata_object


def _xr_spec_iz(stmdata_object):
	"""
	Create a DataSet containing the Lock-In (LIA) and Current spectroscopy data
	Use the absolute values of the tip positions are in the attributes
	"""

	# extract the numpy array containing the Current data from the spym object
	currentarray = stmdata_object.spymdata.Current.data

	# reshape Current data
	currentfw = currentarray[:, 0::2]
	currentbw = currentarray[:, 1::2]

	"""
	Coordinates of the spectroscopy map
	"""
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	# Here we only need the first x and y components
	xcoo = np.array(stmdata_object.spymdata.Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = np.array(stmdata_object.spymdata.Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra.
	tempx = xcoo[0]
	tempy = ycoo[0]

	"""
	Constructing the xarray Dataset 
	"""
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding specific attributes
	xrspec = xr.Dataset(
		data_vars = dict(
			current = (['z', 'repetitions', 'zscandir'], np.stack((currentfw, currentbw), axis=-1)*10**12),
			x = tempx*10**9,
			y = tempy*10**9
			),
		coords = dict(
			z = stmdata_object.spymdata.coords['Current_x'].data*10**9,
			repetitions = np.array(range(stmdata_object.repetitions)),
			zscandir = np.array(['up', 'down'], dtype = 'U')
			),
		attrs = dict(filename = _get_filename(stmdata_object.filename))
	)

	xrspec.attrs['speccoord_x'] = tempx*10**9
	xrspec.attrs['speccoord_y'] = tempy*10**9
	xrspec.attrs['speccoord_x units'] = 'nm'
	xrspec.attrs['speccoord_y units'] = 'nm'
	xrspec.coords['z'].attrs['units'] = 'nm'
	xrspec.coords['z'].attrs['long units'] = 'nanometer'
	xrspec['current'].attrs['units'] = 'pA'
	xrspec['current'].attrs['long units'] = 'picoampere'

	stmdata_object.spectra = xrspec
	return stmdata_object


def _xr_image(stmdata_object):
	# topography
	topofw = stmdata_object.spymdata.Topography_Forward
	topobw = stmdata_object.spymdata.Topography_Backward
	
	# Load image data
	# use spym to align (flatten the data) and planefit
	topofw, bg = align(topofw, baseline='median')
	topobw, bg = align(topobw, baseline='median')
	topofw, bg = plane(topofw)
	topobw, bg = plane(topobw)

	# current
	currfw = stmdata_object.spymdata.Current_Forward
	currbw = stmdata_object.spymdata.Current_Backward
	# lia
	liafw = stmdata_object.spymdata.LIA_Current_Forward
	liabw = stmdata_object.spymdata.LIA_Current_Backward

	# The image data also needs to be flipped along the slow scan direction,
	# so that it shows up as the RHK Rev software would display it, when plotting with xarray.plot().
	# this behaviour is because xarray.plot, uses by default pcolormesh() to plot.
	# pcolormesh() flips the data along the slow scan direction. imshow() plots it the way it looks in RHK Rev and Gwyddion.
	topofw = np.flipud(topofw)
	topobw = np.flipud(topobw)
	currfw = np.flipud(currfw)
	currbw = np.flipud(currbw)
	liafw = np.flipud(liafw)
	liabw = np.flipud(liabw)

	# coordinates
	# absolute values should be found by adding the X Y offsets
	xoff = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Xoffset']
	yoff = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Yoffset']

	# these are the relative coordinates (from 0 to size of the image)
	xx = stmdata_object.spymdata.Topography_Forward_x.data
	yy = stmdata_object.spymdata.Topography_Forward_y.data

	# calculate the relative coordinates, including rotation
	# the offset refers to the corner of the image, so we need to account for that
	xlength = np.abs(xx[-1] - xx[0])
	ylength = np.abs(yy[-1] - yy[0])

	xoff -= xlength/2
	yoff -= ylength/2

	# create xarray Dataset of the image data
	xrimage = xr.Dataset(
		data_vars = dict(
			topography = (['y', 'x', 'scandir'], np.stack((topofw.data, topobw.data), axis=-1)*10**9),
			current = (['y', 'x', 'scandir'], np.stack((currfw.data, currbw.data), axis=-1)*10**12),
			lia = (['y', 'x', 'scandir'], np.stack((liafw.data, liabw.data), axis=-1)*10**12)
			),
		coords = dict(
			x = xx*10**9,
			y = yy*10**9,
			scandir = np.array(['forward', 'backward'])
			),
		attrs = dict(
			filename = _get_filename(stmdata_object.filename),
			xoffset = xoff*10**9,
			yoffset = yoff*10**9,
			xoffset_units = 'nm',
			yoffset_units = 'nm'
			)
		)

	xrimage['topography'].attrs['units'] = 'nm'
	xrimage['topography'].attrs['long units'] = 'nanometer'
	xrimage['lia'].attrs['units'] = 'pA'
	xrimage['lia'].attrs['long units'] = 'picoampere'
	xrimage['current'].attrs['units'] = 'pA'
	xrimage['current'].attrs['long units'] = 'picoampere'
	xrimage.coords['x'].attrs['units'] = 'nm'
	xrimage.coords['y'].attrs['units'] = 'nm'
	xrimage.coords['x'].attrs['long units'] = 'nanometer'
	xrimage.coords['y'].attrs['long units'] = 'nanometer'
	xrimage.coords['x'].attrs['note'] = 'slow scan direction\n'
	xrimage.coords['y'].attrs['note'] = 'fast scan direction\n'

	stmdata_object.image = xrimage
	return stmdata_object


def _add_map_metadata(stmdata_object):
	if stmdata_object.spectype == 'iv':
		stmdata_object.spectra['lia'].attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
		stmdata_object.spectra['lia'].attrs['bias units'] = 'V'
		stmdata_object.spectra['lia'].attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
		stmdata_object.spectra['lia'].attrs['setpoint units'] = 'pA'
		stmdata_object.spectra['lia'].attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']

	stmdata_object.spectra['current'].attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
	stmdata_object.spectra['current'].attrs['bias units'] = 'V'
	stmdata_object.spectra['current'].attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra['current'].attrs['setpoint units'] = 'pA'
	stmdata_object.spectra['current'].attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']
	stmdata_object.spectra.attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
	stmdata_object.spectra.attrs['bias units'] = 'V'
	stmdata_object.spectra.attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra.attrs['setpoint units'] = 'pA'
	stmdata_object.spectra.attrs['measurement date'] = stmdata_object.spymdata.Current.attrs['RHK_Date']
	stmdata_object.spectra.attrs['measurement time'] = stmdata_object.spymdata.Current.attrs['RHK_Time']
	stmdata_object.spectra.attrs['scan angle'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Angle']
	stmdata_object.spectra.attrs['LI amplitude'] = stmdata_object.spymdata.Current.attrs['RHK_CH1Drive_Amplitude']*10**3
	stmdata_object.spectra.attrs['LI amplitude unit'] = 'mV'
	stmdata_object.spectra.attrs['LI frequency'] = stmdata_object.spymdata.Current.attrs['RHK_CH1Drive_Frequency']
	stmdata_object.spectra.attrs['LI frequency unit'] = 'Hz'
	stmdata_object.spectra.attrs['LI phase'] = stmdata_object.spymdata.Current.attrs['RHK_Lockin0_PhaseOffset']

	# store the data and spectrum type in the attributes
	stmdata_object.spectra.attrs['datatype'] = stmdata_object.datatype
	stmdata_object.spectra.attrs['spectype'] = stmdata_object.spectype

	return stmdata_object


def _add_line_metadata(stmdata_object):
	if stmdata_object.spectype == 'iv':
		stmdata_object.spectra['lia'].attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
		stmdata_object.spectra['lia'].attrs['bias units'] = 'V'
		stmdata_object.spectra['lia'].attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
		stmdata_object.spectra['lia'].attrs['setpoint units'] = 'pA'
		stmdata_object.spectra['lia'].attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']

	stmdata_object.spectra['current'].attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
	stmdata_object.spectra['current'].attrs['bias units'] = 'V'
	stmdata_object.spectra['current'].attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra['current'].attrs['setpoint units'] = 'pA'
	stmdata_object.spectra['current'].attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']
	stmdata_object.spectra.attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
	stmdata_object.spectra.attrs['bias units'] = 'V'
	stmdata_object.spectra.attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra.attrs['setpoint units'] = 'pA'
	stmdata_object.spectra.attrs['measurement date'] = stmdata_object.spymdata.Current.attrs['RHK_Date']
	stmdata_object.spectra.attrs['measurement time'] = stmdata_object.spymdata.Current.attrs['RHK_Time']
	stmdata_object.spectra.attrs['scan angle'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Angle']
	stmdata_object.spectra.attrs['LI amplitude'] = stmdata_object.spymdata.Current.attrs['RHK_CH1Drive_Amplitude']*10**3
	stmdata_object.spectra.attrs['LI amplitude unit'] = 'mV'
	stmdata_object.spectra.attrs['LI frequency'] = stmdata_object.spymdata.Current.attrs['RHK_CH1Drive_Frequency']
	stmdata_object.spectra.attrs['LI frequency unit'] = 'Hz'
	stmdata_object.spectra.attrs['LI phase'] = stmdata_object.spymdata.Current.attrs['RHK_Lockin0_PhaseOffset']

	# store the data and spectrum type in the attributes
	stmdata_object.spectra.attrs['datatype'] = stmdata_object.datatype
	stmdata_object.spectra.attrs['spectype'] = stmdata_object.spectype

	return stmdata_object


def _add_spec_metadata(stmdata_object):
	if stmdata_object.spectype == 'iv':
		stmdata_object.spectra['lia'].attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
		stmdata_object.spectra['lia'].attrs['bias units'] = 'V'
		stmdata_object.spectra['lia'].attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
		stmdata_object.spectra['lia'].attrs['setpoint units'] = 'pA'
		stmdata_object.spectra['lia'].attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']
		stmdata_object.spectra.coords['bias'].attrs['units'] = 'V'
		stmdata_object.spectra.coords['bias'].attrs['long units'] = 'volt'
	
	stmdata_object.spectra['current'].attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
	stmdata_object.spectra['current'].attrs['bias units'] = 'V'
	stmdata_object.spectra['current'].attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra['current'].attrs['setpoint units'] = 'pA'
	stmdata_object.spectra['current'].attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']

	stmdata_object.spectra.attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
	stmdata_object.spectra.attrs['bias units'] = 'V'
	stmdata_object.spectra.attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra.attrs['setpoint units'] = 'pA'
	stmdata_object.spectra.attrs['measurement date'] = stmdata_object.spymdata.Current.attrs['RHK_Date']
	stmdata_object.spectra.attrs['measurement time'] = stmdata_object.spymdata.Current.attrs['RHK_Time']
	stmdata_object.spectra.attrs['LI amplitude'] = stmdata_object.spymdata.Current.attrs['RHK_CH1Drive_Amplitude']*10**3
	stmdata_object.spectra.attrs['LI amplitude unit'] = 'mV'
	stmdata_object.spectra.attrs['LI frequency'] = stmdata_object.spymdata.Current.attrs['RHK_CH1Drive_Frequency']
	stmdata_object.spectra.attrs['LI frequency unit'] = 'Hz'
	stmdata_object.spectra.attrs['LI phase'] = stmdata_object.spymdata.Current.attrs['RHK_Lockin0_PhaseOffset']
	
	# store the data and spectrum type in the attributes
	stmdata_object.spectra.attrs['datatype'] = stmdata_object.datatype
	stmdata_object.spectra.attrs['spectype'] = stmdata_object.spectype
	
	return stmdata_object


def _add_image_metadata(stmdata_object):
	stmdata_object.image.attrs['bias'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Bias']
	stmdata_object.image.attrs['bias units'] = 'V'
	stmdata_object.image.attrs['setpoint'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Current']*10**12
	stmdata_object.image.attrs['setpoint units'] = 'pA'
	stmdata_object.image.attrs['measurement date'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Date']
	stmdata_object.image.attrs['measurement time'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Time']
	stmdata_object.image.attrs['scan angle'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Angle']
	
	# store the data and spectrum type in the attributes
	# in the case for a map, the datatype value will not be 'image', but 'map'
	stmdata_object.image.attrs['datatype'] = stmdata_object.datatype
	stmdata_object.image.attrs['spectype'] = stmdata_object.spectype

	return stmdata_object


def load_rhksm4(filename):
	"""Load the data from the .sm4 file using the old loader from spym"""
	return rhksm4.load(filename)

def load_spym(filename):
	"""Load the data from the .sm4 file using spym"""
	return load(filename)

