import xarray as xr
import pylab as pl
import spym
## Using the old loader
from spym.io import rhksm4

class stmdata:
	"""
	stmData class as a container for the xarray based structure of the data
	Parameters:
	repetitions: the number of spectra in each physical position of the tip
	alternate: True if forward and backward bias sweeps are turned on, False if not
	"""
	def __init__(self, filename, repetitions = 1, alternate = True, datatype = 'map', **kwargs):
		# check if parameters passed to the class are valid
		if repetitions <= 0:
			print("repetitions needs to be an integer, with a value of 1 or above. Default is 1")
		elif isinstance(repetitions, int) == False:
			print("repetitions needs to be an integer. Default is 1")

		if isinstance(alternate, bool) == False:
			print("alternate needs to be a bool variable: True or False. Default is True")

		if (datatype != 'map') and (datatype != 'line') and (datatype != 'spec') and (datatype != 'image'):
			print('datatype must be either: map, line, spec or image')
			return


		self.filename = filename
		# number of spectra at a tip position
		self.repetitions = repetitions
		# Boolean value, True if alternate scan directions is turned on
		self.alternate = alternate

		# if the file contains spectroscopy map
		if datatype == 'map':
			self = load_specmap(self)
		elif datatype == 'line':
			self = load_line(self)

	def print_info(self):
		for item in self.__dict__:
			print(item)
		print('\nspymdata:')
		for item in self.spymdata:
			print('\t', item)


def load_specmap(stmdata_object):
	# total number of spectra in one postion of the tip
	stmdata_object.numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions
	
	# Load the data using spym
	stmdata_object.spymdata = load_spym(stmdata_object.filename)
	
	# check software version. Not tested for MinorVer < 6
	l = list(stmdata_object.spymdata.keys())
	if stmdata_object.spymdata[l[-1]].attrs['RHK_MinorVer'] < 6:
		print('stmdatastruct not tested for RHK Rev version < 6. Some things might not work as expected.')
	
	# create lia and current DataArrays
	# rearrange the spectroscopy data into a map
	stmdata_object.lia = lia_xr(stmdata_object)
	stmdata_object.current = current_xr(stmdata_object)

	# rescale the dimensions to nice values
	stmdata_object.lia = rescale_lia(stmdata_object)
	stmdata_object.current = rescale_current(stmdata_object)

	# add metadata to the xarray
	stmdata_object = add_metadata(stmdata_object)

	return stmdata_object

def load_line(stmdata_object):
	# total number of spectra in one postion of the tip
	stmdata_object.numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions
	
	# Load the data using spym
	stmdata_object.spymdata = load_spym(stmdata_object.filename)
	
	# check software version. Not tested for MinorVer < 6
	l = list(stmdata_object.spymdata.keys())
	if stmdata_object.spymdata[l[-1]].attrs['RHK_MinorVer'] < 6:
		print('stmdatastruct not tested for RHK Rev version < 6. Some things might not work as expected.')
	
	# create lia and current DataArrays
	# rearrange the spectroscopy data into a map
	stmdata_object.lia = lia_xr(stmdata_object)
	stmdata_object.current = current_xr(stmdata_object)

	# rescale the dimensions to nice values
	stmdata_object.lia = rescale_lia(stmdata_object)
	stmdata_object.current = rescale_current(stmdata_object)

	# add metadata to the xarray
	stmdata_object = add_metadata(stmdata_object)

	return stmdata_object


def lia_xr(stmdata_object):
	"""
	In spym the spectroscopy data is loaded into an array,
	which has axis=0 the number of datapoints in the spectra
	and axis=1 the number of spectra in total.

	When rearranging, the number of repetitions within each tip position is assumed to be 1
	and alternate scan direction is assumed to be turned on.
	These options can be changed by the parameters, `repetitions` and `alternate`
	"""

	# extract the numpy array containing the LIA data from the spym object
	specarray = stmdata_object.spymdata.LIA_Current.data
	
	# total number of spectra in one postion of the tip
	numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions
	# size of the map in mapsize x mapsize
	mapsize = int(pl.sqrt(specarray.shape[1] / numberofspectra))
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

	"""
	Coordinates of the spectroscopy map
	"""
	# 'RHK_SpecDrift_Xcoord' are the coordinates of the spectra.
	# This contains the coordinates in the order that the spectra are in. 
	xcoo = pl.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = pl.array(stmdata_object.spymdata.LIA_Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra. This is a coordinates mesh
	# at the end slicing the arrays to get the X, Y coordinates, we don't need the mesh
	tempx = pl.reshape(xcoo, (mapsize, mapsize, numberofspectra), order='C')[0, :, 0]
	tempy = pl.reshape(ycoo, (mapsize, mapsize, numberofspectra), order='C')[:, 0, 0]

	# constructing the xarray DataArray for the spectrum
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding DataArray specific attributes
	xrspec = xr.DataArray(pl.stack((liafw, liabw), axis=-1),
                          dims = ['bias', 'specpos_x', 'specpos_y', 'repetitions', 'biasscandir'],
                          coords = dict(
                               bias = stmdata_object.spymdata.coords['LIA_Current_x'].data,
                               specpos_x = tempx,
                               specpos_y = tempy,
                               repetitions = pl.array(range(stmdata_object.repetitions)),
                               biasscandir = pl.array(['left', 'right'], dtype = 'U')
                          ),
                          attrs = dict(
                                   filename = stmdata_object.filename
                          )
                    )
	return xrspec

def current_xr(stmdata_object):
	"""
	In spym the spectroscopy data is loaded into an array,
	which has axis=0 the number of datapoints in the spectra
	and axis=1 the number of spectra in total.

	When rearranging, the number of repetitions within each tip position is assumed to be 1
	and alternate scan direction is assumed to be turned on.
	These options can be changed by the parameters, `repetitions` and `alternate`
	"""

	# extract the numpy array containing the LIA data from the spym object
	currentarray = stmdata_object.spymdata.Current.data
	
	# total number of spectra in one postion of the tip
	numberofspectra = (stmdata_object.alternate + 1)*stmdata_object.repetitions
	# size of the map in mapsize x mapsize
	mapsize = int(pl.sqrt(currentarray.shape[1] / numberofspectra))
	# collect all spectra measured in the same `X, Y` coordinate into an axis (last) of an array.
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
	xcoo = pl.array(stmdata_object.spymdata.Current.attrs['RHK_SpecDrift_Xcoord'])
	ycoo = pl.array(stmdata_object.spymdata.Current.attrs['RHK_SpecDrift_Ycoord'])
	# reshaping the coordinates similarly to the spectra. This is a coordinates mesh
	# at the end slicing the arrays to get the X, Y coordinates, we don't need the mesh
	tempx = pl.reshape(xcoo, (mapsize, mapsize, numberofspectra), order='C')[0, :, 0]
	tempy = pl.reshape(ycoo, (mapsize, mapsize, numberofspectra), order='C')[:, 0, 0]

	# constructing the xarray DataArray for the spectrum
	# stacking the forward and backward bias sweeps and using the scandir coordinate
	# also adding DataArray specific attributes
	xrcurrent = xr.DataArray(pl.stack((currentfw, currentbw), axis=-1),
                          dims = ['bias', 'specpos_x', 'specpos_y', 'repetitions', 'biasscandir'],
                          coords = dict(
                               bias = stmdata_object.spymdata.Current.Current_x.data,
                               specpos_x = tempx,
                               specpos_y = tempy,
                               repetitions = pl.array(range(stmdata_object.repetitions)),
                               biasscandir = pl.array(['left', 'right'], dtype = 'U')
                          ),
                          attrs = dict(
                                   filename = stmdata_object.filename
                          )
                    )
	return xrcurrent

def rescale_lia(stmdata_object):
	"""
	rescale the data to nice values, nm for distances, pA for current and LIA
	"""
	# convert meters to nm
	stmdata_object.lia.coords['specpos_x'] = stmdata_object.lia.coords['specpos_x']*10**9
	stmdata_object.lia.coords['specpos_y'] = stmdata_object.lia.coords['specpos_y']*10**9
	# convert A to pA
	stmdata_object.lia.data = stmdata_object.lia.data*10**12
	stmdata_object.lia.attrs['units'] = 'pA'
	stmdata_object.lia.attrs['long units'] = 'picoampere'
	stmdata_object.lia.coords['specpos_x'].attrs['units'] = 'nm'
	stmdata_object.lia.coords['specpos_y'].attrs['units'] = 'nm'
	stmdata_object.lia.coords['specpos_x'].attrs['long units'] = 'nanometer'
	stmdata_object.lia.coords['specpos_y'].attrs['long units'] = 'nanometer'

	return stmdata_object.lia

def rescale_current(stmdata_object):
	"""
	rescale the data to nice values, nm for distances, pA for current and LIA
	"""
	stmdata_object.current.coords['specpos_x'] = stmdata_object.current.coords['specpos_x']*10**9
	stmdata_object.current.coords['specpos_y'] = stmdata_object.current.coords['specpos_y']*10**9
	stmdata_object.current.data = stmdata_object.current.data*10**12
	stmdata_object.current.attrs['units'] = 'pA'
	stmdata_object.current.attrs['long units'] = 'picoampere'
	stmdata_object.current.coords['specpos_x'].attrs['units'] = 'nm'
	stmdata_object.current.coords['specpos_y'].attrs['units'] = 'nm'
	stmdata_object.current.coords['specpos_x'].attrs['long units'] = 'nanometer'
	stmdata_object.current.coords['specpos_y'].attrs['long units'] = 'nanometer'

	return stmdata_object.current

def add_metadata(stmdata_object):
	stmdata_object.lia.attrs['bias'] = stmdata_object.spymdata.LIA_Current.attrs['bias']
	stmdata_object.current.attrs['bias'] = stmdata_object.spymdata.Current.attrs['bias']

	stmdata_object.lia.attrs['bias units'] = 'V'
	stmdata_object.current.attrs['bias units'] = 'V'

	stmdata_object.lia.attrs['setpoint'] = stmdata_object.spymdata.LIA_Current.attrs['RHK_Current']*10**12
	stmdata_object.current.attrs['setpoint'] = stmdata_object.spymdata.Current.attrs['RHK_Current']*10**12

	stmdata_object.lia.attrs['setpoint units'] = 'pA'
	stmdata_object.current.attrs['setpoint units'] = 'pA'

	stmdata_object.lia.attrs['measurement date'] = stmdata_object.spymdata.LIA_Current.attrs['RHK_Date']
	stmdata_object.current.attrs['measurement date'] = stmdata_object.spymdata.Current.attrs['RHK_Date']
	stmdata_object.lia.attrs['measurement time'] = stmdata_object.spymdata.LIA_Current.attrs['RHK_Time']
	stmdata_object.current.attrs['measurement time'] = stmdata_object.spymdata.Current.attrs['RHK_Time']

	stmdata_object.lia.attrs['time_per_point'] = stmdata_object.spymdata.LIA_Current.attrs['time_per_point']
	stmdata_object.current.attrs['time_per_point'] = stmdata_object.spymdata.Current.attrs['time_per_point']

	return stmdata_object

"""Using spym to load the data from the sm4 file"""
def load_rhksm4(filename):
	"""Load the data from the .sm4 file using the old loader"""
	return rhksm4.load(filename)

def load_spym(filename):
	"""Load the data from the .sm4 file using spym"""
	return spym.load(filename)

