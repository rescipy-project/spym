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
	def __init__(self, filename, repetitions = 1, alternate = True, **kwargs):
		if repetitions <= 0:
			print("repetitions needs to be an integer, with a value of 1 or above. Default is 1")
		elif isinstance(repetitions, int) == False:
			print("repetitions needs to be an integer. Default is 1")

		if isinstance(alternate, bool) == False:
			print("alternate needs to be a bool variable: True or False. Default is True")

		self.filename = filename
		# total number of spectra in one postion of the tip
		self.numberofspectra = (alternate + 1)*repetitions
		
		"""Load the data from rhksm4"""
		self.spymdata = load_spym(filename)
		self.lia_fw, self.lia_bw = rearrange_specmap(self.spymdata.LIA_Current.data)

	def print_info(self):
		for item in self.__dict__:
			print(item)
		print('\nspymdata:')
		for item in self.spymdata:
			print('\t', item)

def rearrange_specmap(specarray, repetitions = 1, alternate = True, **kwargs):
	"""
	In spym the spectroscopy data is loaded into an array,
	which has axis=0 the number of datapoints in the spectra
	and axis=1 the number of spectra in total.

	When rearranging, the number of repetitions within each tip position is assumed to be 1
	and alternate scan direction is assumed to be turned on.
	These options can be changed by the parameters, `repetitions` and `alternate`
	"""

	# specarray is a numpy array
	# total number of spectra in one postion of the tip
	numberofspectra = (alternate + 1)*repetitions
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

	return liafw, liabw


"""Using spym to load the data from the sm4 file"""
def load_rhksm4(filename):
	"""Load the data from the .sm4 file using the old loader"""
	return rhksm4.load(filename)

def load_spym(filename):
	"""Load the data from the .sm4 file using spym"""
	return spym.load(filename)

