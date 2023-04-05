import xarray as xr
import pylab as pl
import numpy as np
import spym
## Using the old loader
from spym.io import rhksm4
## flatten and plane fitting
from spym.process.level import align
from spym.process.level import plane

class stmdata:
	"""
	stmData class as a container for the xarray based structure of the data
	Parameters:
	repetitions: the number of spectra in each physical position of the tip
	alternate: True if forward and backward bias sweeps are turned on, False if not
	"""
	def __init__(self, filename, repetitions = 1, alternate = True, datatype = 'none', **kwargs):
		# check if parameters passed to the class are valid
		if repetitions <= 0:
			print("repetitions needs to be an integer, with a value of 1 or above. Default is 1")
		elif isinstance(repetitions, int) == False:
			print("repetitions needs to be an integer. Default is 1")

		if isinstance(alternate, bool) == False:
			print("alternate needs to be a bool variable: True or False. Default is True")

		self.filename = filename
		# number of spectra at a tip position
		self.repetitions = repetitions
		# Boolean value, True if alternate scan directions is turned on
		self.alternate = alternate

		# Load the data using spym
		self.spymdata = load_spym(self.filename)
		# check software version. Not tested for MinorVer < 6
		l = list(self.spymdata.keys())
		if self.spymdata[l[-1]].attrs['RHK_MinorVer'] < 6:
			print('stmdatastruct not tested for RHK Rev version < 6. Some things might not work as expected.')

		# check type of data contained in the file, if no type is specified
		if datatype == 'none':
			self.datatype, self.spectype = checkdatatype(self)
		else:
			if (datatype != 'map') and (datatype != 'line') and (datatype != 'spec') and (datatype != 'image'):
				print('datatype must be either: map, line, spec or image')
				return
			else:
				self.datatype = datatype

		# load data into xarray, for all data types
		if self.datatype == 'map':
			self = load_specmap(self)
		elif self.datatype == 'line':
			self = load_line(self)
		elif self.datatype == 'spec':
			self = load_spec(self)
		elif self.datatype == 'image':
			self = load_image(self)

	def print_info(self):
		for item in self.__dict__:
			print(item)
		print('\nspymdata:')
		for item in self.spymdata:
			print('\t', item)


def checkdatatype(stmdata_object):
	# Look at the metadata and structure of spectra coordinates to determine the type of file being worked with
	l = list(stmdata_object.spymdata.keys())
	if stmdata_object.spymdata[l[-1]].attrs['RHK_LineType'] == 7:
		stmdata_object.spectype = 'iv'
	elif stmdata_object.spymdata[l[-1]].attrs['RHK_LineType'] == 8:
		stmdata_object.spectype = 'iz'
	elif stmdata_object.spymdata[l[-1]].attrs['RHK_LineType'] == 0:
		stmdata_object.spectype = 'none'
	
	if stmdata_object.spymdata[l[-1]].attrs['RHK_PageType'] == 1:
		stmdata_object.datatype = 'image'
	elif stmdata_object.spymdata[l[-1]].attrs['RHK_PageType'] == 38:
		stmdata_object.datatype = 'spec'
	elif stmdata_object.spymdata[l[-1]].attrs['RHK_PageType'] == 16:
		# this can be either a line spectrum or a map
		# decide based on the aspect ratio of the spectroscopy tip positions
		xcoo = pl.array(stmdata_object.spymdata[l[-1]].attrs['RHK_SpecDrift_Xcoord'])
		ycoo = pl.array(stmdata_object.spymdata[l[-1]].attrs['RHK_SpecDrift_Ycoord'])
		if aspect_ratio(xcoo, ycoo) > 10:
			stmdata_object.datatype = 'line'
		else:
			stmdata_object.datatype = 'map'

	return stmdata_object.datatype, stmdata_object.spectype


def aspect_ratio(x, y):
    xy = np.stack((x, y), axis=0)
    eigvals, eigvecs = np.linalg.eig(np.cov(xy))
    center = xy.mean(axis=-1)
    for val, vec in zip(eigvals, eigvecs.T):
        val *= 2
        xcov,ycov = np.vstack((center + val * vec, center, center - val * vec)).T
    aspect = max(eigvals) / min(eigvals)
    return aspect


def load_specmap(stmdata_object):
	# total number of spectra in one postion of the tip
	stmdata_object.numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions

	# decide if it's a dI/dV or I(z) map
	if stmdata_object.spectype == 'iv':
		# create a DataSet, containing the LIA and Current maps, with appropriate position coordinates
		stmdata_object = xr_map(stmdata_object)
		# rescale the dimensions to nice values
		stmdata_object = rescale_map(stmdata_object)
		# add metadata to the xarray
		stmdata_object = add_map_metadata(stmdata_object)

		# also load the image
		stmdata_object = load_image(stmdata_object)
	elif stmdata_object.spectype == 'iz':
		return
	return stmdata_object


def load_line(stmdata_object):
	# total number of spectra in one postion of the tip
	stmdata_object.numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions

	# decide if it's a dI/dV or I(z) map
	if stmdata_object.spectype == 'iv':
		# create a DataSet, containing the LIA and Current maps, with appropriate position coordinates
		stmdata_object = xr_line(stmdata_object)
		# rescale the dimensions to nice values
		stmdata_object = rescale_line(stmdata_object)
		# add metadata to the xarray
		stmdata_object = add_line_metadata(stmdata_object)

		# also load the image
		stmdata_object = load_image(stmdata_object)
	elif stmdata_object.spectype == 'iz':
		return
	return stmdata_object


def load_spec(stmdata_object):
	# in this case the total number of spectra can be inferred
	# total number of spectra in one postion of the tip
	stmdata_object.repetitions = int(stmdata_object.spymdata.LIA_Current.data.shape[1] / 2)
	stmdata_object.numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions

	# decide if it's a dI/dV or I(z) map
	if stmdata_object.spectype == 'iv':
		# create a DataSet, containing the LIA and Current maps, with appropriate position coordinates
		stmdata_object = xr_spec(stmdata_object)
		# rescale the dimensions to nice values
		stmdata_object = rescale_spec(stmdata_object)
		# add metadata to the xarray
		stmdata_object = add_spec_metadata(stmdata_object)
	elif stmdata_object.spectype == 'iz':
		return
	return stmdata_object


def load_image(stmdata_object):
	# load the image data
	stmdata_object = xr_image(stmdata_object)
	# rescale dimensions
	stmdata_object = rescale_image(stmdata_object)
	# add metadata
	stmdata_object = add_image_metadata(stmdata_object)
	return stmdata_object


def xr_map(stmdata_object):
	"""
	!!!!! ISSUE: need to change spectrum rearranging for the case where alternate is False

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
	mapsize = int(pl.sqrt(specarray.shape[1] / numberofspectra))

	# reshape LIA data
	# collect all spectra measured in the same `X, Y` coordinate into an axis (last) of an array.
	temp = pl.reshape(specarray, (specarray.shape[0], -1, numberofspectra), order='C')
	# Every other spectrum is a forward and backward scan in bias sweep. Separate the forward and backward scans into differing arrays by slicing.
	# These are all the forward and backward bias sweep spectra, arranged along axis=1, with axis=2 being the repetitions
	spec_fw = temp[:, :, 0::2]
	spec_bw = temp[:, :, 1::2]
	# reshape the forward and backward parts into a map
	speccmap_fw = pl.reshape(spec_fw, (spec_fw.shape[0], mapsize, mapsize, spec_fw.shape[2]), order='C')
	speccmap_bw = pl.reshape(spec_bw, (spec_bw.shape[0], mapsize, mapsize, spec_bw.shape[2]), order='C')
	"""
	The last axis (in this case with length of 1) contains the repeated scans in one particular pixel.
	If the `repetitions` variable is set to greater than 1, this will contains the repeated spectra within an `X, Y` pixel.
	The array needs to be flipped along axis = 1 (the "x" axis in the topography image) to fit with the data read by the ASCII method
	"""
	liafw = pl.flip(speccmap_fw, axis=1)
	liabw = pl.flip(speccmap_bw, axis=1)

	# reshape Current data
	temp = pl.reshape(currentarray, (currentarray.shape[0], -1, numberofspectra), order='C')
	# Every other spectrum is a forward and backward scan in bias sweep. Separate the forward and backward scans into differing arrays by slicing.
	# These are all the forward and backward bias sweep spectra, arranged along axis=1, with axis=2 being the repetitions
	current_fw = temp[:, :, 0::2]
	current_bw = temp[:, :, 1::2]
	# reshape the forward and backward parts into a map
	currentmap_fw = pl.reshape(current_fw, (current_fw.shape[0], mapsize, mapsize, current_fw.shape[2]), order='C')
	currentmap_bw = pl.reshape(current_bw, (current_bw.shape[0], mapsize, mapsize, current_bw.shape[2]), order='C')
	"""
	The last axis (in this case with length of 1) contains the repeated scans in one particular pixel.
	If the `repetitions` variable is set to greater than 1, this will contains the repeated spectra within an `X, Y` pixel.
	The array needs to be flipped along axis = 1 (the "x" axis in the topography image) to fit with the data read by the ASCII method
	"""
	currentfw = pl.flip(currentmap_fw, axis=1)
	currentbw = pl.flip(currentmap_bw, axis=1)	

	"""
	Coordinates of the spectroscopy map
	"""
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	xcoo = pl.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = pl.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra. This is a coordinates mesh
	# at the end slicing the arrays to get the X, Y coordinates, we don't need the mesh
	meshx = pl.reshape(xcoo, (mapsize, mapsize, numberofspectra), order='C')[:, :, 0]
	meshy = pl.reshape(ycoo, (mapsize, mapsize, numberofspectra), order='C')[:, :, 0]
	tempx = pl.reshape(xcoo, (mapsize, mapsize, numberofspectra), order='C')[0, :, 0]
	tempy = pl.reshape(ycoo, (mapsize, mapsize, numberofspectra), order='C')[:, 0, 0]

	"""
	Constructing the xarray DataSet 
	"""
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding specific attributes
	xrspec = xr.Dataset(
		data_vars = dict(
			lia = (['bias', 'specpos_x', 'specpos_y', 'repetitions', 'biasscandir'], pl.stack((liafw, liabw), axis=-1)),
			current = (['bias', 'specpos_x', 'specpos_y', 'repetitions', 'biasscandir'], pl.stack((currentfw, currentbw), axis=-1)),
			x = (['specpos_x', 'specpos_y'], meshx*10**9),
			y = (['specpos_x', 'specpos_y'], meshy*10**9)
			),
		coords = dict(
			bias = stmdata_object.spymdata.coords['LIA_Current_x'].data,
			specpos_x = tempx,
			specpos_y = tempy,
			repetitions = pl.array(range(stmdata_object.repetitions)),
			biasscandir = pl.array(['left', 'right'], dtype = 'U')
			),
		attrs = dict(filename = stmdata_object.filename)
	)

	stmdata_object.spectra = xrspec
	return stmdata_object


def xr_line(stmdata_object):
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
	numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions
	# size of the line, the number of the different physical positions of the tip
	linesize = int(specarray.shape[1] / numberofspectra)

	# reshape LIA data
	# Every other spectrum is a forward and backward scan in bias sweep. Separate the forward and backward scans into differing arrays by slicing.
	# These are all the forward and backward bias sweep spectra, arranged along axis=1, with axis=2 being the repetitions
	templia = pl.reshape(specarray, (specarray.shape[0], -1, numberofspectra), order='C')
	liafw = templia[:, :, 0::2]
	liabw = templia[:, :, 1::2]

	# reshape Current data
	tempcurr = pl.reshape(currentarray, (currentarray.shape[0], -1, numberofspectra), order='C')
	currentfw = tempcurr[:, :, 0::2]
	currentbw = tempcurr[:, :, 1::2]

	"""
	Coordinates of the spectroscopy map
	"""
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	xcoo = pl.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = pl.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra. Need only every second coordinate
	tempx = xcoo[0::2]
	tempy = ycoo[0::2]
	linelength = pl.sqrt((tempx[-1] - tempx[0])**2 + (tempy[-1] - tempy[0])**2)
	# distance coordinates along the line in [nm]
	linecoord = pl.linspace(0, linelength, num=tempx.shape[0])*10**9

	"""
	Constructing the xarray Dataset 
	"""
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding specific attributes
	xrspec = xr.Dataset(
		data_vars = dict(
			lia = (['bias', 'dist', 'repetitions', 'biasscandir'], pl.stack((liafw, liabw), axis=-1)),
			current = (['bias', 'dist', 'repetitions', 'biasscandir'], pl.stack((currentfw, currentbw), axis=-1)),
			x = (['dist'], tempx*10**9),
			y = (['dist'], tempy*10**9)
			),
		coords = dict(
			bias = stmdata_object.spymdata.coords['LIA_Current_x'].data,
			dist = linecoord,
			repetitions = pl.array(range(stmdata_object.repetitions)),
			biasscandir = pl.array(['left', 'right'], dtype = 'U')
			),
		attrs = dict(filename = stmdata_object.filename)
	)

	xrspec.coords['dist'].attrs['units'] = 'nm'
	xrspec.coords['dist'].attrs['long units'] = 'nanometer'
	stmdata_object.spectra = xrspec
	return stmdata_object


def xr_spec(stmdata_object):
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

	"""
	Coordinates of the spectroscopy map
	"""
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	# Here we only need the first x and y components
	xcoo = pl.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = pl.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra. Need only every second coordinate
	tempx = xcoo[0]
	tempy = ycoo[0]

	"""
	Constructing the xarray Dataset 
	"""
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding specific attributes
	xrspec = xr.Dataset(
		data_vars = dict(
			lia = (['bias', 'repetitions', 'biasscandir'], pl.stack((liafw, liabw), axis=-1)),
			current = (['bias', 'repetitions', 'biasscandir'], pl.stack((currentfw, currentbw), axis=-1))
			),
		coords = dict(
			bias = stmdata_object.spymdata.coords['LIA_Current_x'].data,
			repetitions = pl.array(range(stmdata_object.repetitions)),
			biasscandir = pl.array(['left', 'right'], dtype = 'U')
			),
		attrs = dict(filename = stmdata_object.filename)
	)

	xrspec.attrs['speccoord_x'] = tempx*10**9
	xrspec.attrs['speccoord_y'] = tempy*10**9
	xrspec.attrs['speccoord_x units'] = 'nm'
	xrspec.attrs['speccoord_y units'] = 'nm'
	stmdata_object.spectra = xrspec
	return stmdata_object


def xr_image(stmdata_object):
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

	# coordinates
	# absolute values should be found by adding the X Y offsets
	xoff = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Xoffset']
	yoff = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Yoffset']

	xx = stmdata_object.spymdata.Topography_Forward_x.data
	yy = stmdata_object.spymdata.Topography_Forward_y.data

	# the offset refers to the corner of the image, so we need to account for that
	xlength = abs(xx[-1] - xx[0])
	ylength = abs(yy[-1] - yy[0])
	xoff -= xlength/2
	yoff -= ylength/2

	"""
	create xarray Dataset of the image data
	"""
	xrimage = xr.Dataset(
		data_vars = dict(
			topography = (['x', 'y', 'scandir'], pl.stack((topofw.data, topobw.data), axis=-1)),
			current = (['x', 'y', 'scandir'], pl.stack((currfw.data, currbw.data), axis=-1)),
			lia = (['x', 'y', 'scandir'], pl.stack((liafw.data, liabw.data), axis=-1))
			),
		coords = dict(
			x = xx + xoff,
			y = yy + yoff,
			scandir = pl.array(['forward', 'backward'])
			),
		attrs = dict(
			filename = stmdata_object.filename,
			xoffset = xoff*10**9,
			yoffset = yoff*10**9,
			xoffset_units = 'nm',
			yoffset_units = 'nm'
			)
		)
	
	stmdata_object.image = xrimage
	return stmdata_object


def rescale_map(stmdata_object):
	"""
	rescale the xarray Dataset
	rescale the data to nice values, nm for distances, pA for current and LIA
	"""
	# convert meters to nm
	stmdata_object.spectra.coords['specpos_x'] = stmdata_object.spectra.coords['specpos_x']*10**9
	stmdata_object.spectra.coords['specpos_y'] = stmdata_object.spectra.coords['specpos_y']*10**9
	# convert A to pA
	stmdata_object.spectra['lia'].data = stmdata_object.spectra['lia'].data*10**12
	stmdata_object.spectra['current'].data = stmdata_object.spectra['current'].data*10**12
	stmdata_object.spectra['lia'].attrs['units'] = 'pA'
	stmdata_object.spectra['lia'].attrs['long units'] = 'picoampere'
	stmdata_object.spectra['current'].attrs['units'] = 'pA'
	stmdata_object.spectra['current'].attrs['long units'] = 'picoampere'
	stmdata_object.spectra.coords['specpos_x'].attrs['units'] = 'nm'
	stmdata_object.spectra.coords['specpos_y'].attrs['units'] = 'nm'
	stmdata_object.spectra.coords['specpos_x'].attrs['long units'] = 'nanometer'
	stmdata_object.spectra.coords['specpos_y'].attrs['long units'] = 'nanometer'
	return stmdata_object


def rescale_line(stmdata_object):
	"""
	rescale the xarray Dataset
	rescale the data to nice values, nm for distances, pA for current and LIA
	"""
	stmdata_object.spectra['x'].attrs['units'] = 'nm'
	stmdata_object.spectra['y'].attrs['units'] = 'nm'
	stmdata_object.spectra['x'].attrs['long units'] = 'nanometer'
	stmdata_object.spectra['y'].attrs['long units'] = 'nanometer'
	# convert A to pA
	stmdata_object.spectra['lia'].data = stmdata_object.spectra['lia'].data*10**12
	stmdata_object.spectra['current'].data = stmdata_object.spectra['current'].data*10**12
	stmdata_object.spectra['lia'].attrs['units'] = 'pA'
	stmdata_object.spectra['lia'].attrs['long units'] = 'picoampere'
	stmdata_object.spectra['current'].attrs['units'] = 'pA'
	stmdata_object.spectra['current'].attrs['long units'] = 'picoampere'
	return stmdata_object


def rescale_spec(stmdata_object):
	"""
	rescale the xarray Dataset
	rescale the data to nice values, nm for distances, pA for current and LIA
	"""
	# convert A to pA
	stmdata_object.spectra['lia'].data = stmdata_object.spectra['lia'].data*10**12
	stmdata_object.spectra['current'].data = stmdata_object.spectra['current'].data*10**12
	stmdata_object.spectra['lia'].attrs['units'] = 'pA'
	stmdata_object.spectra['lia'].attrs['long units'] = 'picoampere'
	stmdata_object.spectra['current'].attrs['units'] = 'pA'
	stmdata_object.spectra['current'].attrs['long units'] = 'picoampere'
	return stmdata_object


def rescale_image(stmdata_object):
	"""
	rescale the xarray Dataset
	rescale the data to nice values, nm for distances, pA for current and LIA
	"""
	# convert meters to nm
	stmdata_object.image.coords['x'] = stmdata_object.image.coords['x']*10**9
	stmdata_object.image.coords['y'] = stmdata_object.image.coords['y']*10**9
	stmdata_object.image['topography'].data = stmdata_object.image['topography'].data*10**9
	stmdata_object.image['topography'].attrs['units'] = 'nm'
	stmdata_object.image['topography'].attrs['long units'] = 'nanometer'
	# convert A to pA
	stmdata_object.image['lia'].data = stmdata_object.image['lia'].data*10**12
	stmdata_object.image['current'].data = stmdata_object.image['current'].data*10**12
	stmdata_object.image['lia'].attrs['units'] = 'pA'
	stmdata_object.image['lia'].attrs['long units'] = 'picoampere'
	stmdata_object.image['current'].attrs['units'] = 'pA'
	stmdata_object.image['current'].attrs['long units'] = 'picoampere'
	stmdata_object.image.coords['x'].attrs['units'] = 'nm'
	stmdata_object.image.coords['y'].attrs['units'] = 'nm'
	stmdata_object.image.coords['x'].attrs['long units'] = 'nanometer'
	stmdata_object.image.coords['y'].attrs['long units'] = 'nanometer'
	return stmdata_object


def add_map_metadata(stmdata_object):
	stmdata_object.spectra['lia'].attrs['bias'] = stmdata_object.spymdata.LIA_Current.attrs['bias']
	stmdata_object.spectra['current'].attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
	stmdata_object.spectra.coords['bias'].attrs['units'] = 'V'
	stmdata_object.spectra.coords['bias'].attrs['long units'] = 'volt'
	stmdata_object.spectra['lia'].attrs['bias units'] = 'V'
	stmdata_object.spectra['current'].attrs['bias units'] = 'V'
	stmdata_object.spectra['lia'].attrs['setpoint'] = stmdata_object.spymdata.LIA_Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra['current'].attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra['lia'].attrs['setpoint units'] = 'pA'
	stmdata_object.spectra['current'].attrs['setpoint units'] = 'pA'
	stmdata_object.spectra['lia'].attrs['time_per_point'] = stmdata_object.spymdata.LIA_Current.attrs['time_per_point']
	stmdata_object.spectra['current'].attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']

	stmdata_object.spectra.attrs['bias'] = stmdata_object.spymdata.LIA_Current.attrs['bias']
	stmdata_object.spectra.attrs['bias units'] = 'V'
	stmdata_object.spectra.attrs['setpoint'] = stmdata_object.spymdata.LIA_Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra.attrs['setpoint units'] = 'pA'
	stmdata_object.spectra.attrs['measurement date'] = stmdata_object.spymdata.Current.attrs['RHK_Date']
	stmdata_object.spectra.attrs['measurement time'] = stmdata_object.spymdata.Current.attrs['RHK_Time']
	return stmdata_object


def add_line_metadata(stmdata_object):
	stmdata_object.spectra['lia'].attrs['bias'] = stmdata_object.spymdata.LIA_Current.attrs['bias']
	stmdata_object.spectra['current'].attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
	stmdata_object.spectra.coords['bias'].attrs['units'] = 'V'
	stmdata_object.spectra.coords['bias'].attrs['long units'] = 'volt'
	stmdata_object.spectra['lia'].attrs['bias units'] = 'V'
	stmdata_object.spectra['current'].attrs['bias units'] = 'V'
	stmdata_object.spectra['lia'].attrs['setpoint'] = stmdata_object.spymdata.LIA_Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra['current'].attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra['lia'].attrs['setpoint units'] = 'pA'
	stmdata_object.spectra['current'].attrs['setpoint units'] = 'pA'
	stmdata_object.spectra['lia'].attrs['time_per_point'] = stmdata_object.spymdata.LIA_Current.attrs['time_per_point']
	stmdata_object.spectra['current'].attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']

	stmdata_object.spectra.attrs['bias'] = stmdata_object.spymdata.LIA_Current.attrs['bias']
	stmdata_object.spectra.attrs['bias units'] = 'V'
	stmdata_object.spectra.attrs['setpoint'] = stmdata_object.spymdata.LIA_Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra.attrs['setpoint units'] = 'pA'
	stmdata_object.spectra.attrs['measurement date'] = stmdata_object.spymdata.Current.attrs['RHK_Date']
	stmdata_object.spectra.attrs['measurement time'] = stmdata_object.spymdata.Current.attrs['RHK_Time']
	return stmdata_object


def add_spec_metadata(stmdata_object):
	stmdata_object.spectra['lia'].attrs['bias'] = stmdata_object.spymdata.LIA_Current.attrs['bias']
	stmdata_object.spectra['current'].attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']
	stmdata_object.spectra.coords['bias'].attrs['units'] = 'V'
	stmdata_object.spectra.coords['bias'].attrs['long units'] = 'volt'
	stmdata_object.spectra['lia'].attrs['bias units'] = 'V'
	stmdata_object.spectra['current'].attrs['bias units'] = 'V'
	stmdata_object.spectra['lia'].attrs['setpoint'] = stmdata_object.spymdata.LIA_Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra['current'].attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra['lia'].attrs['setpoint units'] = 'pA'
	stmdata_object.spectra['current'].attrs['setpoint units'] = 'pA'
	stmdata_object.spectra['lia'].attrs['time_per_point'] = stmdata_object.spymdata.LIA_Current.attrs['time_per_point']
	stmdata_object.spectra['current'].attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']

	stmdata_object.spectra.attrs['bias'] = stmdata_object.spymdata.LIA_Current.attrs['bias']
	stmdata_object.spectra.attrs['bias units'] = 'V'
	stmdata_object.spectra.attrs['setpoint'] = stmdata_object.spymdata.LIA_Current.attrs['RHK_Current']*10**12
	stmdata_object.spectra.attrs['setpoint units'] = 'pA'
	stmdata_object.spectra.attrs['measurement date'] = stmdata_object.spymdata.Current.attrs['RHK_Date']
	stmdata_object.spectra.attrs['measurement time'] = stmdata_object.spymdata.Current.attrs['RHK_Time']
	return stmdata_object


def add_image_metadata(stmdata_object):
	stmdata_object.image.attrs['bias'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Bias']
	stmdata_object.image.attrs['bias units'] = 'V'
	stmdata_object.image.attrs['setpoint'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Current']*10**12
	stmdata_object.image.attrs['setpoint units'] = 'pA'
	stmdata_object.image.attrs['measurement date'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Date']
	stmdata_object.image.attrs['measurement time'] = stmdata_object.spymdata.Topography_Forward.attrs['RHK_Time']
	return stmdata_object


"""Using spym to load the data from the sm4 file"""
def load_rhksm4(filename):
	"""Load the data from the .sm4 file using the old loader"""
	return rhksm4.load(filename)

def load_spym(filename):
	"""Load the data from the .sm4 file using spym"""
	return spym.load(filename)

