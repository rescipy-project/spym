# -*- coding: utf-8 -*-
#
#   Copyright © 2014 - 2019 Stephan Zevenhuizen
#   access2theMatrix, (06-11-2019).
#

'''
    access2theMatrix is a Python library for accessing Scienta Omicron
    (NanoScience) (NanoTechnology) MATRIX Control System result files.
    Scanning Probe Microscopy (SPM) Image data, Single Point Spectroscopy
    (SPS) Curve data, Phase/Amplitude Curve data and volume Continuous Imaging
    Tunneling Spectroscopy (CITS) data will be accessed by this library.
'''

__version__ = '0.4.1'

__all__ = ['Im', 'Cu', 'MtrxData']

from . import access2thematrix

class Im(access2thematrix.Im):
    '''access2theMatrix_image_structure'''

class Cu(access2thematrix.Cu):
    '''access2theMatrix_curve_structure'''

class MtrxData(access2thematrix.MtrxData):
    '''
        The methods to open SPM Image, SPS Curve, Phase/Amplitude Curve and
        volume CITS Curves result files, to select one out of the four possible
        traces (forward/up, backward/up, forward/down and backward/down) for
        images and volume CITS curves, and to select one out of the two possible
        traces (trace, retrace) for spectroscopy curves. Includes method for
        experiment element parameters overview.
    '''

    def open(self, result_data_file):
        '''
        Opens a MATRIX Control System result file.

        Parameters
        ----------
        result_data_file : str
            The pathname of the MATRIX Control System result file.

        Returns
        -------
        traces : dict
            Dictionary of enumerated tracenames for the trace parameter in
            the instancemethod select_image and select_curve, and for the trace
            keys in the volume_scan dictionary object.
        message : str
            Error or succes message of the opening of the file.
        '''
        return access2thematrix.MtrxData.open(self, result_data_file)

    def select_image(self, trace = access2thematrix.MtrxData.ALL_2D_TRACES[0]):
        '''
        The selected image is returned.

        Parameters
        ----------
        trace : str, optional
            Use the traces dictionary, a return from the open method, to set
            this parameter.

        Returns
        -------
        im : access2theMatrix_image_structure
            Returns a Im class (image structure) containing the selected
            image and metadata.
        message : str
            Error or succes message of the image selection.
        '''
        return access2thematrix.MtrxData.select_image(self, trace)

    def select_curve(self, trace = access2thematrix.MtrxData.ALL_1D_TRACES[0]):
        '''
        The selected curve is returned.

        Parameters
        ----------
        trace : str, optional
            Use the traces dictionary, a return from the open method, to set
            this parameter.

        Returns
        -------
        cu : access2theMatrix_curve_structure
            Returns a Cu class (curve structure) containing the selected
            curve and metadata.
        message : str
            Error or succes message of the curve selection.
        '''
        return access2thematrix.MtrxData.select_curve(self, trace)

    def get_experiment_element_parameters(self):
        '''
        Experiment element parameters with their values and units are listed.

        Returns
        -------
        eepas : list
            Returns a list of lists with parameter, value and unit.
        message : str
            A printable sorted text list of the experiment element parameters
            with their values and units.
        '''
        return access2thematrix.MtrxData.get_experiment_element_parameters(self)
