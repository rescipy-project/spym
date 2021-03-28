"""The main SCALA class that represents an Omicron SCALA file.
"""

import numpy as np
import os

class OMICRONscala:
    """This is the main class that represents an Omicron SCALA file
    
    Args:
        filepath: the name of the .par file to be opened
    """

    def __init__(self,
                 filepath):
        """Load data and metadata relative to the given .par file
        
        Args:
                filepath: path to the .par file.
        
        Returns: metadata as dictionary and data as numpy array
        
        """

        self._filepath = filepath
        self._path = os.path.dirname(filepath)

        self._meta = self._loadMeta()
        self._data = self._loadData()

        self._channels = self._addChannels()

    def __getitem__(self, index):
        return self._channels[index]

    def _loadMeta(self):
        """Load metadata from .par file into a dictionary
        
        Returns: dictionary of metadata
        
        """

        # Open .par file and load content
        with open(self._filepath) as f:
            meta = f.readlines()

        # Remove newline character and whitespaces
        meta = [e.strip() for e in meta]
        meta = [e.replace(' ','') for e in meta]

        # Remove final comment part
        meta = [e.split(';',1)[0] for e in meta]

        # Remove empty lines
        meta = [e for e in meta if e not in ('', ';')]

        # Add key to channel parameters, for each channel
        self._chlist = list()
        self._imglist = list()
        self._speclist = list()
        self._SpecParameter = ''

        for i,e in enumerate(meta):
            if 'TopographicChannel' in e:
                chName = meta[i+8][-3:].upper()+"_"
                meta[i] = chName+meta[i]
                meta[i+1] = chName+'Direction:'+meta[i+1]
                meta[i+2] = chName+'MinimumRawValue:'+meta[i+2]
                meta[i+3] = chName+'MaximumRawValue:'+meta[i+3]
                meta[i+4] = chName+'MinimumPhysValue:'+meta[i+4]
                meta[i+5] = chName+'MaximumPhysValue:'+meta[i+5]
                meta[i+6] = chName+'Resolution:'+meta[i+6]
                meta[i+7] = chName+'PhysicalUnit:'+meta[i+7]
                meta[i+8] = chName+'Filename:'+meta[i+8]
                meta[i+9] = chName+'DisplayName:'+meta[i+9]

                self._chlist.append(chName)
                self._imglist.append(chName)

            elif 'SpectroscopyChannel' in e:
                chName = meta[i+16][-3:].upper()+"_"
                meta[i] = chName+meta[i]
                self._SpecParameter = meta[i+1]
                meta[i+1] = chName+'Parameter:'+meta[i+1]
                meta[i+2] = chName+'Direction:'+meta[i+2]
                meta[i+3] = chName+'MinimumRawValue:'+meta[i+3]
                meta[i+4] = chName+'MaximumRawValue:'+meta[i+4]
                meta[i+5] = chName+'MinimumPhysValue:'+meta[i+5]
                meta[i+6] = chName+'MaximumPhysValue:'+meta[i+6]
                meta[i+7] = chName+'Resolution:'+meta[i+7]
                meta[i+8] = chName+'PhysicalUnit:'+meta[i+8]
                meta[i+9] = chName+'NumberSpecPoints:'+meta[i+9]
                meta[i+10] = chName+'StartPoint:'+meta[i+10]
                meta[i+11] = chName+'EndPoint:'+meta[i+11]
                meta[i+12] = chName+'Increment:'+meta[i+12]
                meta[i+13] = chName+'AcqTimePerPoint:'+meta[i+13]
                meta[i+14] = chName+'DelayTimePerPoint:'+meta[i+14]
                meta[i+15] = chName+'Feedback:'+meta[i+15]
                meta[i+16] = chName+'Filename:'+meta[i+16]
                meta[i+17] = chName+'DisplayName:'+meta[i+17]

                self._chlist.append(chName)
                self._speclist.append(chName)

            elif self._SpecParameter+'Parameter' in e:
                meta[i] = 'SpecParam:'+self._SpecParameter
                meta[i+1] = 'SpecParamRampSpeedEnabled:'+meta[i+1]
                meta[i+2] = 'SpecParamT1us:'+meta[i+2]
                meta[i+3] = 'SpecParamT2us:'+meta[i+3]
                meta[i+4] = 'SpecParamT3us:'+meta[i+4]
                meta[i+5] = 'SpecParamT4us:'+meta[i+5]

        # Split list into pairs
        meta = [e.split(':',1) for e in meta]

        # Create dictionary for metadata
        meta = {k:v for k,v in meta}

        # Adjust date as YYYY-MM-DD and time as HH:MM
        year = '20'+meta['Date'][6:8]
        month = meta['Date'][3:5]
        day = meta['Date'][0:2]
        hours = meta['Date'][8:10]
        seconds = meta['Date'][11:13]

        meta['Time'] = hours+':'+seconds
        meta['Date'] = year+'-'+month+'-'+day

        # Calculate timestamp in seconds
        timeStamp = meta['Date']+'T'+meta['Time']+":00"
        meta['Timestamp'] = timeStamp

        return meta

    def _loadData(self):
        """Load data from .par file into a numpy array
        
        Returns: multidimensional numpy array
        
        """

        # Initialize data array
        xsize = int(self._meta['ImageSizeinX'])
        ysize = int(self._meta['ImageSizeinY'])
        data = list()

        # Cycle over image channels
        for i, chPrefix in enumerate(self._imglist):
            chFile = self._meta[chPrefix+'Filename']
            # Load data from current channel
            data.append(np.resize(np.fromfile(os.path.join(self._path, chFile),dtype='>i2'), (xsize, ysize)))

        # Return data
        return data

    def _addChannels(self):

        channels = list()

        for i, chName in enumerate(self._imglist):

            data = self._data[i]
            attrs = dict()
            for k,v in self._meta.items():
                if chName in k:
                    key = k.replace(chName, '')
                    if key in ['MinimumRawValue', 'MaximumRawValue', 'NumberSpecPoints']:
                        attrs[key] = int(v)
                    elif key in ['MinimumPhysValue', 'MaximumPhysValue', 'Resolution',
                                 'StartPoint', 'EndPoint', 'Increment', 'AcqTimePerPoint:', 'DelayTimePerPoint']:
                        attrs[key] = float(v)
                    else:
                        attrs[key] = v
            attrs.pop("Filename")

            channel = OMICRONchannel(data, {**attrs, **self._globAttrs()})
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

class OMICRONchannel:
    
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
