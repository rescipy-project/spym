import numpy as np
import os

class NANONISsxm:
    """This is the main class that represents a Nanonis sxm file
    
    Args:
        filepath: the name of the .sxm file to be opened
    """

    def __init__(self,
                 filepath):
        """Load data and metadata relative to the given .sxm file
        
        Args:
                filepath: path to the .sxm file.
        
        Returns: metadata as dictionary and data as numpy array
        
        """

        self._file = open(filepath, 'rb')
        self._filepath = filepath
        self._path = os.path.dirname(filepath)

        self._meta = self._loadMeta()
        self._data = self._loadData()

        #self._channels = self._addChannels()

        self._file.close()

    def __getitem__(self, index):
        return self._channels[index]

    def _loadMeta(self):
        """Load metadata from .sxm file into a dictionary
        
        Returns: dictionary of metadata
        
        """

        # Open .sxm file and load metadata
        meta = list()
        while True:
            l = self._file.readline().decode()
            if l == ':SCANIT_END:\n':
                break
            elif l == '\n':
                continue
            meta.append(l)

        # Format metadata dictionary
        meta = "".join(meta)
        meta = dict(item.split(":\n") for item in meta.split("\n:"))
        meta['NANONIS_VERSION'] = meta.pop(':NANONIS_VERSION')
        meta['Scan>channels'] = meta['Scan>channels'].split(";")
        self._data_info = self._parse_meta_block(meta['DATA_INFO'])
        meta['DATA_INFO'] = self._data_info
        meta['Z-CONTROLLER'] = self._parse_meta_block(meta['Z-CONTROLLER'])

        # Seek to data offset, 4 bytes ahead from actual position
        self._file.seek(4, 1)

        return meta

    def _parse_meta_block(self, entry):
        entry_list = list()
        for item in entry.split("\n"):
            if item == "":
                continue
            if item[0] == "\t":
                item = item[1:]
            entry_list.append(item.split("\t"))
        meta_list = list(dict(zip(entry_list[0], values)) for values in entry_list[1:])
        if len(meta_list) == 1:
            return meta_list[0]
        else:
            return meta_list

    def _loadData(self):
        """Load data from .sxm file into a numpy array
        
        Returns: multidimensional numpy array
        
        """

        # Initialize data array
        xsize = int(self._meta['Scan>pixels/line'])
        ysize = int(self._meta['Scan>lines'])
        data = list()

        # Cycle over image channels
        for channel in self._data_info:
            data.append(np.fromfile(self._file, dtype='>f4', count=(xsize*ysize)).reshape((xsize, ysize)))
            if channel['Direction'] == 'both':
                data.append(np.fromfile(self._file, dtype='>f4', count=(xsize*ysize)).reshape((xsize, ysize)))

        # Return data
        return data

    def _addChannels(self):

        channels = list()

        for i, chDict in enumerate(self._data_info):

            data = self._data[i]
            attrs = dict()
            for k,v in self._meta.items():
                if key in ['MinimumRawValue', 'MaximumRawValue', 'NumberSpecPoints']:
                    attrs[key] = int(v)
                elif key in ['MinimumPhysValue', 'MaximumPhysValue', 'Resolution']:
                    attrs[key] = float(v)
                else:
                    attrs[key] = v

            channel = SXMchannel(data, {**attrs, **self._globAttrs()})
            channels.append(channel)

        return channels

    def _globAttrs(self):

        attrs = self._meta.copy()

        for i, chName in enumerate(self._chlist):
            for k,v in self._meta.items():
                if chName in k:
                    del attrs[k]

        float_keys = ['FieldXSizeinnm',
                      'FieldYSizeinnm',
                      'IncrementX',
                      'IncrementY',
                      'ScanAngle',
                      'XOffset',
                      'YOffset',
                      'GapVoltage',
                      'FeedbackSet',
                      'LoopGain',
                      'XResolution',
                      'YResolution',
                      'ScanSpeed',
                      'XDrift',
                      'YDrift',
                      'TopographyTimeperPoint',
                      'ZSpeed',
                      'ZOutputGain',
                      'ZInputGain']
        int_keys = ['Format',
                    'ImageSizeinX',
                    'ImageSizeinY',
                    'SpectroscopyGridValueinX',
                    'SpectroscopyGridValueinY',
                    'SpectroscopyPointsinX',
                    'SpectroscopyLinesinY',
                    'SpecParamT1us',
                    'SpecParamT2us',
                    'SpecParamT3us',
                    'SpecParamT4us']

        for k,v in attrs.items():
            if k in float_keys:
                attrs[k] = float(v)
            elif k in int_keys:
                attrs[k] = int(v)

        return attrs

class SXMchannel:
    
    def __init__(self, data, attrs):
        
        self.data = data
        self.attrs = attrs

        if self.attrs['TopographicChannel'] == "Z":
            channel = "Topography"
        elif self.attrs['TopographicChannel'] == "I":
            channel = "Current"
        else:
            channel = self.attrs['TopographicChannel']
        self.label = channel + "_" + self.attrs['Direction']

        xsize = self.attrs['ImageSizeinX']
        xres = self.attrs['IncrementX']
        ysize = self.attrs['ImageSizeinY']
        yres = self.attrs['IncrementY']
        self.coords = [('y', yres * np.arange(ysize, dtype=np.float)),
                       ('x', xres * np.arange(xsize, dtype=np.float))]
