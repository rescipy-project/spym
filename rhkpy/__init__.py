from .rhkpy_loader import *
from .rhkpy_process import *

hvplot.extension('bokeh')
from holoviews import opts
opts.defaults(
    opts.Image(
        aspect = 1
        )
    ) # can't set default colormap with this