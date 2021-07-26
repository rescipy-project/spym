import matplotlib.pyplot as plt
import hvplot.xarray

class Plotting():
    ''' Plotting.
    
    '''

    def __init__(self, spym_instance):
        self._spym = spym_instance

    def plot(self, title=None, **kwargs):
        ''' Plot data with custom parameters using matplotlib.

        Args:
            title: (optional) title of the figure (string). By default gives some basic information on the data plotted. Pass an empty string to disable it.
            **kwargs: any argument accepted by xarray.plot() function.

        '''

        dr = self._spym._dr
        attrs = dr.attrs

        # Clear plt
        plt.clf()

        # Set plot properties
        if attrs['interpretation'] == 'spectrum':
            # plot wraps matplotlib.pyplot.plot()
            plot = dr.plot.line(hue="y", **kwargs)

        elif attrs['interpretation'] == 'image':
            # plot is an instance of matplotlib.collections.QuadMesh
            plot = dr.plot.pcolormesh(**kwargs)
            fig = plot.get_figure()
            ax = plot.axes
            ax.invert_yaxis()
            # Fit figure pixel size to image
            fig_width, fig_height = self._fit_figure_to_image(fig, dr.data, ax)
            fig.set_size_inches(fig_width, fig_height)

            # Apply colormap
            plot.set_cmap('afmhot')

        else:
            # Create figure
            # xarray plot() wraps:
            #   - matplotlib.pyplot.plot() for 1d arrays
            #   - matplotlib.pyplot.pcolormesh() for 2d arrays
            #   - matplotlib.pyplot.hist() for anything else
            plot = dr.plot(**kwargs)

        # Set figure title
        if title is None:
            title = self._format_title()
        plt.title(title)

        plt.plot()

        return plot

    def hvplot(self, title=None, **kwargs):
        ''' Plot data with custom parameters using hvplot.

        Args:
            title: (optional) title of the figure (string). By default gives some basic information on the data plotted. Pass an empty string to disable it.
            **kwargs: any argument accepted by hvplot() function.

        '''

        dr = self._spym._dr
        attrs = dr.attrs

        # Set figure title
        if title is None:
            title = self._format_title()

        # Set hvplot properties
        if attrs['interpretation'] == 'spectrum':
            hvplot = dr.hvplot(**kwargs).opts(title=title)

        elif attrs['interpretation'] == 'image':
            hvplot = dr.hvplot(**kwargs).opts(title=title,
                                              cmap='afmhot',
                                              frame_width=512,
                                              frame_height=512,
                                              invert_yaxis=True,
                                              data_aspect=1)

        else:
            hvplot = dr.hvplot(**kwargs).opts(title=title)

        return hvplot

    def _format_title(self):
        ''' Provide a title from the metadata of the DataArray.

        '''

        title = ""
        attrs = self._spym._dr.attrs

        if "filename" in attrs:
            title += attrs["filename"] + "\n"

        title += "{:.2f} {}, {:.2f} {}".format(
            attrs["bias"],
            attrs["bias_units"],
            attrs["setpoint"],
            attrs["setpoint_units"])

        return title

    def _fit_figure_to_image(self, figure, image, axis=None):
        ''' Calculate figure size so that plot (matplotlib axis) pixel size is equal to the image size.

        Args:
            figure: matplotlib Figure instance.
            image: 2d numpy array.
            axis: axis of the figure to adapt, if None takes the first (or only) axis.

        Returns:
            adapted width and height of the figure in inches.

        '''

        if axis is None:
            axis = figure.axes[0]
        bounds = axis.bbox.bounds

        im_width, im_height = image.shape

        width_scale = im_width/bounds[2]
        height_scale = im_height/bounds[3]

        fig_width, fig_height = figure.get_size_inches()

        return fig_width*width_scale, fig_height*height_scale
