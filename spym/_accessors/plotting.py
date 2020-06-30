import matplotlib as mpl
import matplotlib.pyplot as plt

class SpymPlotting():
    ''' Plotting.
    
    '''

    def __init__(self, spym_instance):
        self._spym = spym_instance

    def plot(self, title=None, **kwargs):
        ''' Plot data with custom parameters.
        
        Args:
            title: title of the figure (string).
            **kwargs: any argument accepted by xarray.plot() function.
        '''

        # Get plot (matplotlib axes) dimension for the given image
        fig_width, fig_height = mpl.rcParams['figure.figsize']
        fig = self._spym._dr.plot(cmap = 'afmhot')
        ax = fig.axes
        bounds = ax.bbox.bounds

        # Calculate figure size so that plot (matplotlib axes) size is equal to the image size
        im_width, im_height = self._spym._dr.data.shape
        width_scale = im_width/bounds[2]
        height_scale = im_height/bounds[3]
        fig.get_figure().set_size_inches(fig_width*width_scale, fig_height*height_scale)

        # Apply image name title
        if title:
            ax.set_title(title)
