"""The main SM4 class that represents a RHK SM4 file,
with all the necessary attributes and methods
to read the binary data of the .sm4 files (RHK technology Inc.).
"""

import numpy as np
from enum import Enum

## Definition of types

## Object type
class object_type(Enum):
    RHK_OBJECT_UNDEFINED            = 0
    RHK_OBJECT_PAGE_INDEX_HEADER    = 1
    RHK_OBJECT_PAGE_INDEX_ARRAY     = 2
    RHK_OBJECT_PAGE_HEADER          = 3
    RHK_OBJECT_PAGE_DATA            = 4
    RHK_OBJECT_IMAGE_DRIFT_HEADER   = 5
    RHK_OBJECT_IMAGE_DRIFT          = 6
    RHK_OBJECT_SPEC_DRIFT_HEADER    = 7
    RHK_OBJECT_SPEC_DRIFT_DATA      = 8
    RHK_OBJECT_COLOR_INFO           = 9
    RHK_OBJECT_STRING_DATA          = 10
    RHK_OBJECT_TIP_TRACK_HEADER     = 11
    RHK_OBJECT_TIP_TRACK_DATA       = 12
    RHK_OBJECT_PRM                  = 13
    RHK_OBJECT_THUMBNAIL            = 14
    RHK_OBJECT_PRM_HEADER           = 15
    RHK_OBJECT_THUMBNAIL_HEADER     = 16
    RHK_OBJECT_API_INFO             = 17
    RHK_OBJECT_HISTORY_INFO         = 18
    RHK_OBJECT_PIEZO_SENSITIVITY    = 19
    RHK_OBJECT_FREQUENCY_SWEEP_DATA = 20
    RHK_OBJECT_SCAN_PROCESSOR_INFO  = 21
    RHK_OBJECT_PLL_INFO             = 22
    RHK_OBJECT_CH1_DRIVE_INFO       = 23
    RHK_OBJECT_CH2_DRIVE_INFO       = 24
    RHK_OBJECT_LOCKIN0_INFO         = 25
    RHK_OBJECT_LOCKIN1_INFO         = 26
    RHK_OBJECT_ZPI_INFO             = 27
    RHK_OBJECT_KPI_INFO             = 28
    RHK_OBJECT_AUX_PI_INFO          = 29
    RHK_OBJECT_LOWPASS_FILTER0_INFO = 30
    RHK_OBJECT_LOWPASS_FILTER1_INFO = 31

## Page Data type
class page_data_type(Enum):
    RHK_DATA_IMAGE          = 0
    RHK_DATA_LINE           = 1
    RHK_DATA_XY_DATA        = 2
    RHK_DATA_ANNOTATED_LINE = 3
    RHK_DATA_TEXT           = 4
    RHK_DATA_ANNOTATED_TEXT = 5
    RHK_DATA_SEQUENTIAL     = 6
    RHK_DATA_MOVIE          = 7

## Page Source type
class page_source_type(Enum):
    RHK_SOURCE_RAW        = 0
    RHK_SOURCE_PROCESSED  = 1
    RHK_SOURCE_CALCULATED = 2
    RHK_SOURCE_IMPORTED   = 3

## Page type
class page_type(Enum):
    RHK_PAGE_UNDEFINED                   = 0
    RHK_PAGE_TOPOGRAPHIC                 = 1
    RHK_PAGE_CURRENT                     = 2
    RHK_PAGE_AUX                         = 3
    RHK_PAGE_FORCE                       = 4
    RHK_PAGE_SIGNAL                      = 5
    RHK_PAGE_FFT_TRANSFORM               = 6
    RHK_PAGE_NOISE_POWER_SPECTRUM        = 7
    RHK_PAGE_LINE_TEST                   = 8
    RHK_PAGE_OSCILLOSCOPE                = 9
    RHK_PAGE_IV_SPECTRA                  = 10
    RHK_PAGE_IV_4x4                      = 11
    RHK_PAGE_IV_8x8                      = 12
    RHK_PAGE_IV_16x16                    = 13
    RHK_PAGE_IV_32x32                    = 14
    RHK_PAGE_IV_CENTER                   = 15
    RHK_PAGE_INTERACTIVE_SPECTRA         = 16
    RHK_PAGE_AUTOCORRELATION             = 17
    RHK_PAGE_IZ_SPECTRA                  = 18
    RHK_PAGE_4_GAIN_TOPOGRAPHY           = 19
    RHK_PAGE_8_GAIN_TOPOGRAPHY           = 20
    RHK_PAGE_4_GAIN_CURRENT              = 21
    RHK_PAGE_8_GAIN_CURRENT              = 22
    RHK_PAGE_IV_64x64                    = 23
    RHK_PAGE_AUTOCORRELATION_SPECTRUM    = 24
    RHK_PAGE_COUNTER                     = 25
    RHK_PAGE_MULTICHANNEL_ANALYSER       = 26
    RHK_PAGE_AFM_100                     = 27
    RHK_PAGE_CITS                        = 28
    RHK_PAGE_GPIB                        = 29
    RHK_PAGE_VIDEO_CHANNEL               = 30
    RHK_PAGE_IMAGE_OUT_SPECTRA           = 31
    RHK_PAGE_I_DATALOG                   = 32
    RHK_PAGE_I_ECSET                     = 33
    RHK_PAGE_I_ECDATA                    = 34
    RHK_PAGE_I_DSP_AD                    = 35
    RHK_PAGE_DISCRETE_SPECTROSCOPY_PP    = 36
    RHK_PAGE_IMAGE_DISCRETE_SPECTROSCOPY = 37
    RHK_PAGE_RAMP_SPECTROSCOPY_RP        = 38
    RHK_PAGE_DISCRETE_SPECTROSCOPY_RP    = 39

## Line type
class line_type(Enum):
    RHK_LINE_NOT_A_LINE                     = 0
    RHK_LINE_HISTOGRAM                      = 1
    RHK_LINE_CROSS_SECTION                  = 2
    RHK_LINE_LINE_TEST                      = 3
    RHK_LINE_OSCILLOSCOPE                   = 4
    RHK_LINE_RESERVED                       = 5
    RHK_LINE_NOISE_POWER_SPECTRUM           = 6
    RHK_LINE_IV_SPECTRUM                    = 7
    RHK_LINE_IZ_SPECTRUM                    = 8
    RHK_LINE_IMAGE_X_AVERAGE                = 9
    RHK_LINE_IMAGE_Y_AVERAGE                = 10
    RHK_LINE_NOISE_AUTOCORRELATION_SPECTRUM = 11
    RHK_LINE_MULTICHANNEL_ANALYSER_DATA     = 12
    RHK_LINE_RENORMALIZED_IV                = 13
    RHK_LINE_IMAGE_HISTOGRAM_SPECTRA        = 14
    RHK_LINE_IMAGE_CROSS_SECTION            = 15
    RHK_LINE_IMAGE_AVERAGE                  = 16
    RHK_LINE_IMAGE_CROSS_SECTION_G          = 17
    RHK_LINE_IMAGE_OUT_SPECTRA              = 18
    RHK_LINE_DATALOG_SPECTRUM               = 19
    RHK_LINE_GXY                            = 20
    RHK_LINE_ELECTROCHEMISTRY               = 21
    RHK_LINE_DISCRETE_SPECTROSCOPY          = 22
    RHK_LINE_DATA_LOGGER                    = 23
    RHK_LINE_TIME_SPECTROSCOPY              = 24
    RHK_LINE_ZOOM_FFT                       = 25
    RHK_LINE_FREQUENCY_SWEEP                = 26
    RHK_LINE_PHASE_ROTATE                   = 27
    RHK_LINE_FIBER_SWEEP                    = 28

## Image type
class image_type(Enum):
    RHK_IMAGE_NORMAL         = 0
    RHK_IMAGE_AUTOCORRELATED = 1

## Scan direction type
class scan_type(Enum):
    RHK_SCAN_RIGHT = 0
    RHK_SCAN_LEFT  = 1
    RHK_SCAN_UP    = 2
    RHK_SCAN_DOWN  = 3

## Drift option type
class drift_option_type(Enum):
    RHK_DRIFT_DISABLED      = 0
    RHK_DRIFT_EACH_SPECTRA  = 1
    RHK_DRIFT_EACH_LOCATION = 2

## SM4 class definition
class RHKsm4:
    """This is the main class that represents a RHK SM4 file
    
    Args:
        filename: the name of the .sm4 file to be opened
    """

    def __init__(self,
                 filename):

        ## Open the file
        self._file = open(filename, 'rb')

        ## Read the File Header
        self._header = RHKFileHeader(self)
        ## Read Object list of File Header
        self._header._read_object_list(self)

        ## Read Page Index Header
        self._page_index_header = RHKPageIndexHeader(self)
        ## Read Object list of Page Index Header
        self._page_index_header._read_object_list(self)

        ## Seek to the start position of the Page Index Array
        offset = self._page_index_header._get_offset('RHK_OBJECT_PAGE_INDEX_ARRAY')
        self._seek(offset, 0)

        ## Read Page Index Array
        self._pages = []
        for i in range(self._page_index_header.page_count):
            page = RHKPage(self)
            #Read Page Index
            self._pages.append(page)
            #Read Object list of Page Index
            page._read_object_list(self)

        ## Read Pages content
        for page in self:
            page._read()

        ## Close the file
        self._file.close()

        return

    def __getitem__(self, index):
        return self._pages[index]

    def _readb(self, dtype, count):
        '''Read bytewise a single line of the file
        '''

        return np.fromfile(self._file, dtype=dtype, count=count)[0]
    
    def _reads(self, count):
        '''Read bytewise *count* lines of the file and join as string
        '''

        string = ''.join([chr(i) for i in np.fromfile(self._file, dtype=np.uint16, count=count)])
        return string.rstrip('\x00')

    def _readstr(self):
        '''Read RHK string object
        
        Each string is written to file by first writing the string length(2 bytes),
        then the string. So when we read, first read a short value, which gives the
        string length, then read that much bytes which represents the string.
        '''

        length = self._readb(np.uint16, 1)#first 2 bytes is the string length
        string = ''.join([chr(i) for i in np.fromfile(self._file, dtype=np.uint16, count=length)])
        return string.rstrip('\x00')

    def _readtime(self):
        '''Read RHK filetime object
        
        It is expressed in Windows epoch, a 64-bit value representing
        the number of 100-nanosecond intervals since January 1, 1601 (UTC).
        '''

        return np.fromfile(self._file, dtype=np.uint64, count=1)[0]

    def _seek(self, offset, whence):
        '''Seek the file to the given position
        '''

        self._file.seek(offset, whence)

class RHKObject:
    '''Define an RHK object.
    
    An Object contains:
    Object ID: (4 bytes) Type of data stored
    Offset: (4 bytes) Data offset
    Size: (4 bytes) size of the data
    Using the data offset and size, we can read the corresponding object data.
    '''

    def __init__(self, sm4):
        '''Read the object properties.
        '''

        self.id = sm4._readb(np.uint32, 1)
        try:
            self.name = object_type(self.id).name
        except ValueError:
            self.name = 'RHK_OBJECT_UNKNOWN'
        self.offset = sm4._readb(np.uint32, 1)
        self.size = sm4._readb(np.uint32, 1)

        ''' Seek to the end position of the current Object
        (for compatibility with future file versions
        in case Object Field Size is no longer 12 bytes)'''
        #sm4._seek(sm4._header.object_field_size - 12, 1)

class RHKObjectContainer:
    '''Represents a class containing RHK Objects
    '''

    def _read_object_list(self, sm4):
        '''Populate Object list
        '''

        self._object_list = []
        for i in range(self._object_list_count):
            self._object_list.append(RHKObject(sm4))

    def _get_offset(self, object_name):
        '''Get offset of the given object
        '''

        for obj in self._object_list:
            if obj.name == object_name:
                return obj.offset

    def _read_object_content(self, obj):

        # Chech if object position is valid then read it
        if obj.offset != 0 and obj.size != 0:
            if obj.id == 5:
                self._read_ImageDriftHeader(obj.offset)
            elif obj.id == 6:
                self._read_ImageDrift(obj.offset)
            elif obj.id == 7:
                self._read_SpecDriftHeader(obj.offset)
            elif obj.id == 8:
                self._read_SpecDriftData(obj.offset)
            elif obj.id == 9:
                ## Color Info is skipped
                #self._read_ColorInfo(obj.offset)
                pass
            elif obj.id == 10:
                self._read_StringData(obj.offset)
            elif obj.id == 11:
                self._read_TipTrackHeader(obj.offset)
            elif obj.id == 12:
                self._read_TipTrackData(obj.offset)
            elif obj.id == 13:
                # PRMdata is read within _read_PRMHeader()
                #self._read_PRMdata(obj.offset)
                pass
            elif obj.id == 15:
                self._read_PRMHeader(obj.offset)
            elif obj.id == 17:
                self._read_APIInfo(obj.offset)
            elif obj.id == 18:
                self._read_HistoryInfo(obj.offset)
            elif obj.id == 19:
                self._read_PiezoSensitivity(obj.offset)
            elif obj.id == 20:
                self._read_FrequencySweepData(obj.offset)
            elif obj.id == 21:
                self._read_ScanProcessorInfo(obj.offset)
            elif obj.id == 22:
                self._read_PLLInfo(obj.offset)
            elif obj.id == 23:
                self._read_ChannelDriveInfo(obj.offset, 'RHK_CH1Drive')
            elif obj.id == 24:
                self._read_ChannelDriveInfo(obj.offset, 'RHK_CH2Drive')
            elif obj.id == 25:
                self._read_LockinInfo(obj.offset, 'RHK_Lockin0')
            elif obj.id == 26:
                self._read_LockinInfo(obj.offset, 'RHK_Lockin1')
            elif obj.id == 27:
                self._read_PIControllerInfo(obj.offset, 'RHK_ZPI')
            elif obj.id == 28:
                self._read_PIControllerInfo(obj.offset, 'RHK_KPI')
            elif obj.id == 29:
                self._read_PIControllerInfo(obj.offset, 'RHK_AuxPI')
            elif obj.id == 30:
                self._read_LowPassFilterInfo(obj.offset, 'RHK_LowPassFilter0')
            elif obj.id == 31:
                self._read_LowPassFilterInfo(obj.offset, 'RHK_LowPassFilter1')

    def _read_StringData(self, offset):
        ''' Read String Data for the current page.
        
        _string_count gives the number of strings in the current page.
        '''

        self._sm4._seek(offset, 0)

        # Create string labels list, adding any additional (at date unknown) label
        strList = ["RHK_Label",
                   "RHK_SystemText",
                   "RHK_SessionText",
                   "RHK_UserText",
                   "RHK_FileName",
                   "RHK_Date",
                   "RHK_Time",
                   "RHK_Xunits",
                   "RHK_Yunits",
                   "RHK_Zunits",
                   "RHK_Xlabel",
                   "RHK_Ylabel",
                   "RHK_StatusChannelText",
                   "RHK_CompletedLineCount",
                   "RHK_OverSamplingCount",
                   "RHK_SlicedVoltage",
                   "RHK_PLLProStatus",
                   "RHK_SetpointUnit",
                   "CHlist"]
        for i in range(self._string_count - 19):
            strList.append('RHK_Unknown'+"{:0>3d}".format(i))

        # Actual read of the strings
        for k in range(self._string_count):
            if k == 4: #file path
                self._path = self._sm4._readstr()
                self.attrs[strList[k]] = self._sm4._file.name
            elif k in [13, 14]: #conversion to integer
                self.attrs[strList[k]] = int(self._sm4._readstr())
            elif k == 18: # parse CHDriveValues string
                CHlist = self._sm4._readstr().split("\n")
                for i, CH in enumerate(CHlist):
                    self.attrs["RHK_CH"+str(i+1)+"DriveValue"] = float(CH.split(" ")[3])
                    self.attrs["RHK_CH"+str(i+1)+"DriveValueUnits"] = CH.split(" ")[4]
            else:
                self.attrs[strList[k]] = self._sm4._readstr()

        # Create ISO8601 datetime stamp
        mm, dd, yy = self.attrs['RHK_Date'].split('/')
        datetime = '20' + yy + '-' + mm + '-' + dd + 'T' + self.attrs['RHK_Time'] + '.000'
        self.attrs['RHK_DateTime'] = datetime

        # Add line type units based on line_type enum class
        line_type_xunits = {'RHK_LINE_NOT_A_LINE': '',
                            'RHK_LINE_HISTOGRAM': '',
                            'RHK_LINE_CROSS_SECTION': '',
                            'RHK_LINE_LINE_TEST': '',
                            'RHK_LINE_OSCILLOSCOPE': '',
                            'RHK_LINE_RESERVED': '',
                            'RHK_LINE_NOISE_POWER_SPECTRUM': '',
                            'RHK_LINE_IV_SPECTRUM': 'Bias',
                            'RHK_LINE_IZ_SPECTRUM': 'Z',
                            'RHK_LINE_IMAGE_X_AVERAGE': '',
                            'RHK_LINE_IMAGE_Y_AVERAGE': '',
                            'RHK_LINE_NOISE_AUTOCORRELATION_SPECTRUM': '',
                            'RHK_LINE_MULTICHANNEL_ANALYSER_DATA': '',
                            'RHK_LINE_RENORMALIZED_IV': '',
                            'RHK_LINE_IMAGE_HISTOGRAM_SPECTRA': '',
                            'RHK_LINE_IMAGE_CROSS_SECTION': '',
                            'RHK_LINE_IMAGE_AVERAGE': '',
                            'RHK_LINE_IMAGE_CROSS_SECTION_G': '',
                            'RHK_LINE_IMAGE_OUT_SPECTRA': '',
                            'RHK_LINE_DATALOG_SPECTRUM': '',
                            'RHK_LINE_GXY': '',
                            'RHK_LINE_ELECTROCHEMISTRY': '',
                            'RHK_LINE_DISCRETE_SPECTROSCOPY': '',
                            'RHK_LINE_DATA_LOGGER': '',
                            'RHK_LINE_TIME_SPECTROSCOPY': 'Time',
                            'RHK_LINE_ZOOM_FFT': '',
                            'RHK_LINE_FREQUENCY_SWEEP': '',
                            'RHK_LINE_PHASE_ROTATE': '',
                            'RHK_LINE_FIBER_SWEEP': ''}

        if self.attrs['RHK_Xlabel'] == '':
            self.attrs['RHK_Xlabel'] = line_type_xunits[self.attrs['RHK_LineTypeName']]

    def _read_SpecDriftHeader(self, offset):
        ''' Read Spec Drift Header for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_SpecDrift_Filetime'] = self._sm4._readtime()
        self.attrs['RHK_SpecDrift_DriftOptionType'] = self._sm4._readb(np.uint32, 1)
        self.attrs['RHK_SpecDrift_DriftOptionTypeName'] = drift_option_type(self.attrs['RHK_SpecDrift_DriftOptionType']).name
        _ = self._sm4._readb(np.uint32, 1) # SpecDrift StringCount
        self.attrs['RHK_SpecDrift_Channel'] = self._sm4._readstr()

    def _read_SpecDriftData(self, offset):
        ''' Read Spec Drift Data for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_SpecDrift_Time'] = []
        self.attrs['RHK_SpecDrift_Xcoord'] = []
        self.attrs['RHK_SpecDrift_Ycoord'] = []
        self.attrs['RHK_SpecDrift_dX'] = []
        self.attrs['RHK_SpecDrift_dY'] = []
        self.attrs['RHK_SpecDrift_CumulativeX'] = []
        self.attrs['RHK_SpecDrift_CumulativeY'] = []

        for k in range(self.attrs['RHK_Ysize']):
            self.attrs['RHK_SpecDrift_Time'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_SpecDrift_Xcoord'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_SpecDrift_Ycoord'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_SpecDrift_dX'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_SpecDrift_dY'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_SpecDrift_CumulativeX'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_SpecDrift_CumulativeY'].append(self._sm4._readb(np.float32, 1))

    def _read_ImageDriftHeader(self, offset):
        ''' Read Image Drift Header for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_ImageDrift_Filetime'] = self._sm4._readtime()
        self.attrs['RHK_ImageDrift_DriftOptionType'] = self._sm4._readb(np.uint32, 1)
        self.attrs['RHK_ImageDrift_DriftOptionTypeName'] = drift_option_type(self.attrs['RHK_ImageDrift_DriftOptionType']).name

    def _read_ImageDrift(self, offset):
        ''' Read Image Drift for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_ImageDrift_Time'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_ImageDrift_dX'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_ImageDrift_dY'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_ImageDrift_CumulativeX'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_ImageDrift_CumulativeY'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_ImageDrift_VectorX'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_ImageDrift_VectorY'] = self._sm4._readb(np.float32, 1)

    def _read_ColorInfo(self, offset):
        ''' Read Color Info for the current page.
        
        Color Info is only for use into RHK DAW software.
        '''

        self._sm4._seek(offset, 0)

        ## Initialize metadata
        self.attrs['RHK_Color_StructSize'] = []
        self.attrs['RHK_Color_Reserved'] = []

        #HSVColor
        self.attrs['RHK_Color_Hstart'] = []
        self.attrs['RHK_Color_Sstart'] = []
        self.attrs['RHK_Color_Vstart'] = []
        self.attrs['RHK_Color_Hstop'] = []
        self.attrs['RHK_Color_Sstop'] = []
        self.attrs['RHK_Color_Vstop'] = []

        self.attrs['RHK_Color_ClrDirection'] = []
        self.attrs['RHK_Color_NumEntries'] = []
        self.attrs['RHK_Color_StartSlidePos'] = []
        self.attrs['RHK_Color_EndSlidePos'] = []

        #Color Transform
        self.attrs['RHK_Color_Gamma'] = []
        self.attrs['RHK_Color_Alpha'] = []
        self.attrs['RHK_Color_Xstart'] = []
        self.attrs['RHK_Color_Xstop'] = []
        self.attrs['RHK_Color_Ystart'] = []
        self.attrs['RHK_Color_Ystop'] = []
        self.attrs['RHK_Color_MappingMode'] = []
        self.attrs['RHK_Color_Invert'] = []

        for k in range(self._color_info_count):

            self.attrs['RHK_Color_StructSize'].append(self._sm4._readb(np.uint16, 1))
            self.attrs['RHK_Color_Reserved'].append(self._sm4._readb(np.uint16, 1))

            ## HSVColor
            self.attrs['RHK_Color_Hstart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Sstart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Vstart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Hstop'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Sstop'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Vstop'].append(self._sm4._readb(np.float32, 1))

            self.attrs['RHK_Color_ClrDirection'].append(self._sm4._readb(np.uint32, 1))
            self.attrs['RHK_Color_NumEntries'].append(self._sm4._readb(np.uint32, 1))
            self.attrs['RHK_Color_StartSlidePos'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_EndSlidePos'].append(self._sm4._readb(np.float32, 1))

            ## Color Transform
            self.attrs['RHK_Color_Gamma'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Alpha'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Xstart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Xstop'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Ystart'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_Ystop'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_Color_MappingMode'].append(self._sm4._readb(np.uint32, 1))
            self.attrs['RHK_Color_Invert'].append(self._sm4._readb(np.uint32, 1))

    def _read_TipTrackHeader(self, offset):
        ''' Read Tip track Header for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_TipTrack_Filetime'] = self._sm4._readtime()
        self.attrs['RHK_TipTrack_FeatureHeight'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_TipTrack_FeatureWidth'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_TipTrack_TimeConstant'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_TipTrack_CycleRate'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_TipTrack_PhaseLag'] = self._sm4._readb(np.float32, 1)
        _ = self._sm4._readb(np.uint32, 1) # TipTrack StringCount
        self.attrs['RHK_TipTrack_TipTrackInfoCount'] = self._sm4._readb(np.uint32, 1)
        self.attrs["RHK_TipTrack_Channel"] = self._sm4._readstr()

    def _read_TipTrackData(self, offset):
        ''' Read Tip Track Data for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_TipTrack_CumulativeTime'] = []
        self.attrs['RHK_TipTrack_Time'] = []
        self.attrs['RHK_TipTrack_dX'] = []
        self.attrs['RHK_TipTrack_dY'] = []

        for k in range(self.attrs['RHK_TipTrack_TipTrackInfoCount']):
            self.attrs['RHK_TipTrack_CumulativeTime'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_TipTrack_Time'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_TipTrack_dX'].append(self._sm4._readb(np.float32, 1))
            self.attrs['RHK_TipTrack_dY'].append(self._sm4._readb(np.float32, 1))

    def _read_PRMdata(self, offset):
        ''' Read PRM Data for the current page.
        
        Valid only for RHK XPMPro generated files.
        PRM data could be compressed with Zlib.
        '''

        import zlib

        self._sm4._seek(offset, 0)

        if self._PRM_CompressionFlag == 0:
            PRMdata = np.fromfile(self._sm4._file, dtype=np.uint32, count=self._PRM_DataSize)
        elif self._PRM_CompressionFlag == 1:
            comprPRMdata = np.fromfile(self._sm4._file, dtype=np.uint32, count=self._PRM_CompressionSize)
            PRMdata = zlib.decompress(comprPRMdata, wbits=0, bufsize=self._PRM_DataSize)

        self.attrs['RHK_PRMdata'] = PRMdata.decode('CP437')

    def _read_PRMHeader(self, offset):
        ''' Read PRM Header for the current page.
        
        Valid only for RHK XPMPro generated files.
        '''

        self._sm4._seek(offset, 0)

        self._PRM_CompressionFlag = self._sm4._readb(np.uint32, 1)
        self._PRM_DataSize = self._sm4._readb(np.uint32, 1)
        self._PRM_CompressionSize = self._sm4._readb(np.uint32, 1)

        prm_data_offset = self._sm4._header._get_offset('RHK_OBJECT_PRM')
        self._read_PRMdata(prm_data_offset)

    def _read_APIInfo(self, offset):
        ''' Read API Info for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_API_VoltageHigh'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_API_VoltageLow'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_API_Gain'] = self._sm4._readb(np.float32, 1)
        self.attrs['RHK_API_Offset'] = self._sm4._readb(np.float32, 1)

        self.attrs['RHK_API_RampMode'] = self._sm4._readb(np.uint32, 1)
        self.attrs['RHK_API_RampType'] = self._sm4._readb(np.uint32, 1)
        self.attrs['RHK_API_Step'] = self._sm4._readb(np.uint32, 1)
        self.attrs['RHK_API_ImageCount'] = self._sm4._readb(np.uint32, 1)
        self.attrs['RHK_API_DAC'] = self._sm4._readb(np.uint32, 1)
        self.attrs['RHK_API_MUX'] = self._sm4._readb(np.uint32, 1)
        self.attrs['RHK_API_STMBias'] = self._sm4._readb(np.uint32, 1)

        _ = self._sm4._readb(np.uint32, 1) # API StringCount

        self.attrs['RHK_API_Units'] = self._sm4._readstr()

    def _read_HistoryInfo(self, offset):
        ''' Read History Info for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        _ = self._sm4._readb(np.uint32, 1) # History StringCount
        _ = self._sm4._readstr() # History Path
        _ = self._sm4._readstr() # History Pixel2timeFile

    def _read_PiezoSensitivity(self, offset):
        ''' Read Piezo Sensitivity for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_PiezoSensitivity_TubeX'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PiezoSensitivity_TubeY'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PiezoSensitivity_TubeZ'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PiezoSensitivity_TubeZOffset'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PiezoSensitivity_ScanX'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PiezoSensitivity_ScanY'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PiezoSensitivity_ScanZ'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PiezoSensitivity_Actuator'] = self._sm4._readb(np.float64, 1)

        _ = self._sm4._readb(np.uint32, 1) # PiezoSensitivity StringCount

        self.attrs['RHK_PiezoSensitivity_TubeXUnit'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_TubeYUnit'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_TubeZUnit'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_TubeZOffsetUnit'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_ScanXUnit'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_ScanYUnit'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_ScanZUnit'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_ActuatorUnit'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_TubeCalibration'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_ScanCalibration'] = self._sm4._readstr()
        self.attrs['RHK_PiezoSensitivity_ActuatorCalibration'] = self._sm4._readstr()

    def _read_FrequencySweepData(self, offset):
        ''' Read Frequency Sweep Data for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_FrequencySweep_PSDTotalSignal'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_FrequencySweep_PeakFrequency'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_FrequencySweep_PeakAmplitude'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_FrequencySweep_DriveAmplitude'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_FrequencySweep_Signal2DriveRatio'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_FrequencySweep_QFactor'] = self._sm4._readb(np.float64, 1)

        _ = self._sm4._readb(np.uint32, 1) # FrequencySweep StringCount

        self.attrs['RHK_FrequencySweep_TotalSignalUnit'] = self._sm4._readstr()
        self.attrs['RHK_FrequencySweep_PeakFrequencyUnit'] = self._sm4._readstr()
        self.attrs['RHK_FrequencySweep_PeakAmplitudeUnit'] = self._sm4._readstr()
        self.attrs['RHK_FrequencySweep_DriveAmplitudeUnit'] = self._sm4._readstr()
        self.attrs['RHK_FrequencySweep_Signal2DriveRatioUnit'] = self._sm4._readstr()
        self.attrs['RHK_FrequencySweep_QFactorUnit'] = self._sm4._readstr()

    def _read_ScanProcessorInfo(self, offset):
        ''' Read Scan Processor Info for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs['RHK_ScanProcessor_XSlopeCompensation'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_ScanProcessor_YSlopeCompensation'] = self._sm4._readb(np.float64, 1)

        _ = self._sm4._readb(np.uint32, 1) # ScanProcessor StringCount

        self.attrs['RHK_ScanProcessor_XSlopeCompensationUnit'] = self._sm4._readstr()
        self.attrs['RHK_ScanProcessor_YSlopeCompensationUnit'] = self._sm4._readstr()

    def _read_PLLInfo(self, offset):
        ''' Read PLL Info for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        _ = self._sm4._readb(np.uint32, 1) # PLL StringCount
        self.attrs['RHK_PLL_AmplitudeControl'] = self._sm4._readb(np.uint32, 1)

        self.attrs['RHK_PLL_DriveAmplitude'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_DriveRefFrequency'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_LockinFreqOffset'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_LockinHarmonicFactor'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_LockinPhaseOffset'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_PIGain'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_PIIntCutoffFreq'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_PILowerBound'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_PIUpperBound'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_DissPIGain'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_DissPIIntCutoffFreq'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_DissPILowerBound'] = self._sm4._readb(np.float64, 1)
        self.attrs['RHK_PLL_DissPIUpperBound'] = self._sm4._readb(np.float64, 1)

        self.attrs['RHK_PLL_LockinFilterCutoffFreq'] = self._sm4._readstr()

        self.attrs['RHK_PLL_DriveAmplitudeUnit'] = self._sm4._readstr()
        self.attrs['RHK_PLL_DriveFrequencyUnit'] = self._sm4._readstr()
        self.attrs['RHK_PLL_LockinFreqOffsetUnit'] = self._sm4._readstr()
        self.attrs['RHK_PLL_LockinPhaseUnit'] = self._sm4._readstr()
        self.attrs['RHK_PLL_PIGainUnit'] = self._sm4._readstr()
        self.attrs['RHK_PLL_PIICFUnit'] = self._sm4._readstr()
        self.attrs['RHK_PLL_PIOutputUnit'] = self._sm4._readstr()
        self.attrs['RHK_PLL_DissPIGainUnit'] = self._sm4._readstr()
        self.attrs['RHK_PLL_DissPIICFUnit'] = self._sm4._readstr()
        self.attrs['RHK_PLL_DissPIOutputUnit'] = self._sm4._readstr()

    def _read_ChannelDriveInfo(self, offset, metaString):
        ''' Read Channel Drive Info for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        _ = self._sm4._readb(np.uint32, 1) # ChannelDrive StringCount
        self.attrs[metaString + '_MasterOscillator'] = self._sm4._readb(np.uint32, 1)

        self.attrs[metaString + '_Amplitude'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_Frequency'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_PhaseOffset'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_HarmonicFactor'] = self._sm4._readb(np.float64, 1)

        self.attrs[metaString + '_AmplitudeUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_FrequencyUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_PhaseOffsetUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_ReservedUnit'] = self._sm4._readstr()

    def _read_LockinInfo(self, offset, metaString):
        ''' Read Lockin Info for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        _ = self._sm4._readb(np.uint32, 1) # LockinInfo StringCount
        self.attrs[metaString + '_NonMasterOscillator'] = self._sm4._readb(np.uint32, 1)

        self.attrs[metaString + '_Frequency'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_HarmonicFactor'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_PhaseOffset'] = self._sm4._readb(np.float64, 1)

        self.attrs[metaString + '_FilterCutoffFrequency'] = self._sm4._readstr()

        self.attrs[metaString + '_FreqUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_PhaseUnit'] = self._sm4._readstr()

    def _read_PIControllerInfo(self, offset, metaString):
        ''' Read PI Controller Info for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        self.attrs[metaString + '_SetPoint'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_ProportionalGain'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_IntegralGain'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_LowerBound'] = self._sm4._readb(np.float64, 1)
        self.attrs[metaString + '_UpperBound'] = self._sm4._readb(np.float64, 1)

        _ = self._sm4._readb(np.uint32, 1) # PIController StringCount

        self.attrs[metaString + '_FeedbackType'] = self._sm4._readstr()
        self.attrs[metaString + '_SetPointUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_ProportionalGainUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_IntegralGainUnit'] = self._sm4._readstr()
        self.attrs[metaString + '_OutputUnit'] = self._sm4._readstr()

    def _read_LowPassFilterInfo(self, offset, metaString):
        ''' Read Low-Pass Filter Info for the current page.
        
        '''

        self._sm4._seek(offset, 0)

        _ = self._sm4._readb(np.uint32, 1) # LowPassFilter StringCount
        freq, units = self._sm4._readstr().split(" ")
        self.attrs[metaString + '_CutoffFrequency'] = float(freq)
        self.attrs[metaString + '_CutoffFrequencyUnits'] = units

class RHKFileHeader(RHKObjectContainer):
    '''Class representing the File Header.
    
    The File Header contains the general information about the SM4 file
    and the file offset to other details like index header, PRM data etc.
    '''

    def __init__(self, sm4):
        '''Read the File Header.
        
        File header size: (2 bytes) the size for the actual File Header (in current version =56 bytes)
        File header content:
        Signature: (18x2 bytes) "STiMage 005.006 1". Mayor version.Minor version Unicode=1
        Total page count: (4 bytes) the basic structure is a page, where data is saved
        Object list count: (4 bytes) the count of Objects stored just after the file header (currently =3).
        Object field size: (4 bytes) the size of the Object structure (currently =12 bytes per Object)
        Reserved: (4x2 bytes) 2 fields reserved for future use.
        '''

        ## File Header Size
        self.header_size = sm4._readb(np.uint16, 1)

        ## File Header
        self.signature = sm4._reads(18)
        self.total_page_count = sm4._readb(np.uint32, 1)
        self._object_list_count = sm4._readb(np.uint32, 1)
        self.object_field_size = sm4._readb(np.uint32, 1)
        self.reserved = sm4._readb(np.uint32, 2)

        ''' Seek to the end position of the File Header
        (for compatibility with future file versions
        in case File Header Size is no longer 56 bytes)'''
        sm4._seek(self.header_size + 2, 0)

class RHKPageIndexHeader(RHKObjectContainer):
    '''Class representing the Page Index Header.
    '''

    def __init__(self, sm4):
        '''Read the Page Index Header.
        
        Page Index Header content:
        Page count: (4 bytes) Stores the number of pages in the Page Index Array
        Object List Count: Stores the count of Objects stored after Page Index Header (currently =1)
        Reserved: (4x2 bytes) 2 fields reserved for future use.
        Object List: Stores the Objects in the Page Index Header. Currently is stored one Object:
            1. Page Index Array
        '''

        ## Seek to the position of the Page Index Header
        self.offset = sm4._header._get_offset('RHK_OBJECT_PAGE_INDEX_HEADER')
        sm4._seek(self.offset, 0)

        self.page_count = sm4._readb(np.uint32, 1)# the number of pages in the page index array
        self._object_list_count = sm4._readb(np.uint32, 1)
        self.reserved = sm4._readb(np.uint32, 2)

class RHKPageHeader(RHKObjectContainer):
    ''' Class representing the Page Header
    '''

    def __init__(self, page, sm4):
        '''Read the Page Header
        
        The page header stores the header details of each page.
        It is followed by its Objects in the number given by 'object-list_count'.
        '''

        self.sm4 = sm4

        ## Seek for the position of the Page Header
        self.offset = page._get_offset('RHK_OBJECT_PAGE_HEADER')
        self.sm4._seek(self.offset, 0)

        if ( page._page_data_type == 6 ):#"Sequential" Page Data type
            self.read_sequential_type(page)
        else:
            self.read_default_type(page)#all other Page Data types

    def read_sequential_type(self, page):

        page.attrs['RHK_DataType'] = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_DataLength'] = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_ParamCount'] = self.sm4._readb(np.uint32, 1)

        self._object_list_count = self.sm4._readb(np.uint32, 1)

        page.attrs['RHK_DataInfoSize'] = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_DataInfoStringCount'] = self.sm4._readb(np.uint32, 1)

        ## Adding manually these attributes for consistency with subsequent code
        page._page_type = 0
        page._line_type = 0
        page.attrs['RHK_PageType'] = page._page_type
        page.attrs['RHK_PageTypeName'] = page_type(page._page_type).name
        page.attrs['RHK_LineType'] = page._line_type
        page.attrs['RHK_LineTypeName'] = line_type(page._line_type).name
        page._page_data_size = page.attrs['RHK_ParamCount']*(page.attrs['RHK_DataLength'] + 1)

    def read_default_type(self, page):

        _ = self.sm4._readb(np.uint16, 1) # FieldSize
        page._string_count = self.sm4._readb(np.uint16, 1)

        page._page_type = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_PageType'] = page._page_type
        try:
            page.attrs['RHK_PageTypeName'] = page_type(page._page_type).name
        except ValueError:
            page.attrs['RHK_PageTypeName'] = 'RHK_PAGE_UNKNOWN'

        page.attrs['RHK_DataSubSource'] = self.sm4._readb(np.uint32, 1)

        page._line_type = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_LineType'] = page._line_type
        try:
            page.attrs['RHK_LineTypeName'] = line_type(page._line_type).name
        except ValueError:
            page.attrs['RHK_LineTypeName'] = 'RHK_LINE_UNKNOWN'

        page.attrs['RHK_Xcorner'] = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_Ycorner'] = self.sm4._readb(np.uint32, 1)

        ''' Xsize is the number of pixels in the X direction for an image page,
        or the number of points per spectra/line for line pages.'''
        page.attrs['RHK_Xsize'] = self.sm4._readb(np.uint32, 1)

        ''' Ysize is the number of pixels in the Y direction for an image page,
        or the number of spectra stored in the page.'''
        page.attrs['RHK_Ysize'] = self.sm4._readb(np.uint32, 1)

        page._image_type = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_ImageType'] = page._image_type
        try:
            page.attrs['RHK_ImageTypeName'] = image_type(page._image_type).name
        except ValueError:
            page.attrs['RHK_ImageTypeName'] = 'RHK_IMAGE_UNKNOWN'

        page._scan_type = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_ScanType'] = page._scan_type
        try:
            page.attrs['RHK_ScanTypeName'] = scan_type(page._scan_type).name
        except ValueError:
            page.attrs['RHK_ScanTypeName'] = 'RHK_SCAN_UNKNOWN'

        page.attrs['RHK_GroupId'] = self.sm4._readb(np.uint32, 1)

        page._page_data_size = self.sm4._readb(np.uint32, 1)

        page.attrs['RHK_MinZvalue'] = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_MaxZvalue'] = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_Xscale'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_Yscale'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_Zscale'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_XYscale'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_Xoffset'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_Yoffset'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_Zoffset'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_Period'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_Bias'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_Current'] = self.sm4._readb(np.float32, 1)
        page.attrs['RHK_Angle'] = self.sm4._readb(np.float32, 1)

        page._color_info_count = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_GridXsize'] = self.sm4._readb(np.uint32, 1)
        page.attrs['RHK_GridYsize'] = self.sm4._readb(np.uint32, 1)

        self._object_list_count = self.sm4._readb(np.uint32, 1)
        page._32bit_data_flag = self.sm4._readb(np.uint8, 1)
        page._reserved_flags = self.sm4._readb(np.uint8, 3)# 3 bytes
        page._reserved = self.sm4._readb(np.uint8, 60)# 60 bytes

    def read_objects(self, page):

        # Read Page Header objects
        self._read_object_list(self.sm4)

        # Add Data Info if "Sequential" Page Data type
        if ( page._page_data_type == 6 ):

            ## Initialize metadata
            page.attrs['RHK_Sequential_ParamGain'] = []
            page.attrs['RHK_Sequential_ParamLabel'] = []
            page.attrs['RHK_Sequential_ParamUnit'] = []

            for i in range(page.attrs['RHK_ParamCount']):

                ## Parameter gain
                page.attrs['RHK_Sequential_ParamGain'].append(self.sm4._readb(np.float32, 1))
                ## Name of the parameter
                page.attrs['RHK_Sequential_ParamLabel'].append(self.sm4._readstr())
                ## Unit of the parameter
                page.attrs['RHK_Sequential_ParamUnit'].append(self.sm4._readstr())

        # Read each object and add to Page metadata
        for obj in self._object_list:
            page._read_object_content(obj)

class RHKPage(RHKObjectContainer):
    ''' Class representing Page
    '''

    def __init__(self, sm4):
        '''Read the Page Index
        
        Content:
            Page ID: Unique GUID for each Page
            Page Data Type: The type of data stored with the page.
            Page Source Type: Identifies the page source type.
            Object List Count: Stores the count of Objects stored after each Page Index
            Minor Version: (4 bytes) stores the minor version of the file (2 in QP,
                4 in XPMPro, 6 in Rev9)
            Object List: Stores the Objects in the Page Index. Currently we are storing:
                1. Page Header
                2. Page Data
                3. Thumbnail
                4. Thumbnail header
        '''

        self._sm4 = sm4

        ## Initialize Page Index and Page meta dictionaries
        self.attrs = {}
        self.attrs['RHK_PRMdata'] = ""

        self.attrs['RHK_PageID'] = sm4._readb(np.uint16, 8)

        self._page_data_type = sm4._readb(np.uint32, 1)
        self.attrs['RHK_PageDataType'] = self._page_data_type
        try:
            self.attrs['RHK_PageDataTypeName'] = page_data_type(self._page_data_type).name
        except ValueError:
            self.attrs['RHK_PageDataTypeName'] = 'RHK_DATA_UNKNOWN'

        self._page_source_type = sm4._readb(np.uint32, 1)
        self.attrs['RHK_PageSourceType'] = self._page_source_type
        try:
            self.attrs['RHK_PageSourceTypeName'] = page_source_type(self._page_source_type).name
        except ValueError:
            self.attrs['RHK_PageSourceTypeName'] = 'RHK_SOURCE_UNKNOWN'

        self._object_list_count = sm4._readb(np.uint32, 1)
        self.attrs['RHK_MinorVer'] = sm4._readb(np.uint32, 1)

        ## Add signature from File Header
        self.attrs['RHK_Signature'] = sm4._header.signature

    def _read(self):
        '''Read the Page Header and Page Data
        
        Thumbnail and Thumbnail Header are discarded
        '''

        ## Read Page Header
        self._header = RHKPageHeader(self, self._sm4)
        self._header.read_objects(self)

        ## Set page label
        if self.attrs['RHK_PageDataType'] == 0 and self._page_data_type != 6:
            if self.attrs['RHK_ScanType'] == 0:
                scan_direction = '_Forward'
            elif self.attrs['RHK_ScanType'] == 1:
                scan_direction = '_Backward'
        else:
            scan_direction = ''

        if self.attrs['RHK_Label'] != '':
            label = self.attrs['RHK_Label']
            label = label.replace(" ", "_")
            label = label.replace("-", "_")
            if label.startswith("_"):
                label = label[1:]
            self.label = label + scan_direction
        else:
            self.label = "ID" + str(self.attrs['RHK_PageID'])

        ## Read Page Data
        self._read_data()
        
        ## Read PRM data from file header
        for obj in self._sm4._header._object_list:
            self._read_object_content(obj)

    def _read_data(self):
        '''Read Page Data
        '''

        ## Seek for the position of the Page Data
        offset = self._get_offset('RHK_OBJECT_PAGE_DATA')
        self._sm4._seek(offset, 0)

        ## Load data, selecting float or long integer type
        data_size = int(self._page_data_size / 4)
        if ( self._line_type in [1, 6, 9, 10, 11, 13, 18, 19, 21, 22] or self._page_data_type == 6 ):
            raw_data = np.fromfile(self._sm4._file, dtype=np.float32, count=data_size)
            ## For Sequential_data page, the page data contains an array of size ‘n’ with ‘m’ elements is
            ## stored. Where m is the Param count and n is the Data length (array size) stored in the
            ## page header. The first float data in each element represents the output values.
        else:
            raw_data = np.fromfile(self._sm4._file, dtype=np.int32, count=data_size)

        # Reshape and store data
        self.data, self.coords = self._reshape_data(raw_data)

    def _reshape_data(self, raw_data):
        '''Reshape data of the page and create its coordinates
        '''

        xsize = self.attrs['RHK_Xsize']
        ysize = self.attrs['RHK_Ysize']
        xscale = self.attrs['RHK_Xscale']
        yscale = self.attrs['RHK_Yscale']

        # Reshape data
        if self._page_data_type == 0: # Image type

            data = raw_data.reshape(xsize, ysize)

            coords = [('y', abs(yscale) * np.arange(ysize, dtype=np.float64)),
                      ('x', abs(xscale) * np.arange(xsize, dtype=np.float64))]

            # Check scale and adjust accordingly data orientation
            if xscale < 0:
                data = np.flip(data, axis=1)
            if yscale > 0:
                data = np.flip(data, axis=0)

        elif self._page_data_type == 1: # Line type

            data = raw_data.reshape(ysize, xsize)

            xoffset = self.attrs['RHK_Xoffset']
            coords = [('y', int(yscale) * np.arange(ysize, dtype=np.uint32)),
                      ('x', xscale * np.arange(xsize, dtype=np.float64) + xoffset)]

            if self._line_type == 22: # Discrete spectroscopy has shape xsize*(ysize+1)
                tmp = raw_data.reshape(xsize, ysize+1).transpose()
                coords[1] = ('x', tmp[0])
                data = tmp[1:]

        else:

            data = raw_data
            coords = [('x', np.arange(xsize*ysize, dtype=np.uint32))]

        return data, coords
