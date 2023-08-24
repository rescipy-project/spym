from .rhkpy_loader import *
from .rhkpy_process import *

import hvplot.xarray
import holoviews as hv
from holoviews import dim, opts

import logging
from bokeh.util import logconfig
# Suppress Bokeh's warnings
logconfig.basicConfig(level=logging.ERROR)

import warnings
# suppress holoviews warnings
warnings.filterwarnings('ignore', category = UserWarning, module = 'holoviews.plotting.bokeh.plot')

# select bokeh
hvplot.extension('bokeh')

# global plotting options
from holoviews import opts
opts.defaults(
    opts.Image(
        aspect = 'equal'
        )
    ) # can't set default colormap with this