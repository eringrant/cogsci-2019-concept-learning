import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import seaborn as sns; sns.set_style("whitegrid", {'axes.grid': False})
import tensorflow as tf


class TensorBoardPlot(object):

    def __init__(self, name, max_outputs=1, num=0, figsize=(4, 4), dpi=600):
        """TODO

        Args:
            name: name of TF summary

        Returns:
        """
        # plt params
        self.num = num
        self.figsize = figsize
        self.dpi = dpi

        # TF setup
        fig = self._get_fig()
        self.placeholder = tf.placeholder(tf.uint8, (None,) + self._render(fig, expand=False).shape)
        self.summary = tf.summary.image(name, self.placeholder, max_outputs=max_outputs)

    def _get_fig(self):
        plt.clf()
        return plt.figure(num=self.num, figsize=self.figsize, dpi=self.dpi)

    def _render(self, fig, expand=True, axis=False):
        if axis is False:
            plt.axis('off')
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
        plt.close()

        return np.fromstring(buf, dtype=np.uint8).reshape(shape)

    def plot(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")


class TensorBoardActivationHeatmap(TensorBoardPlot):

    def plot(self, data, zoom_factor=None, vmin=None, vmax=None):
        assert len(data.shape) == 3, "Data must have three dimensions (batch index, height, and width)"

        rendered_list = []
        for i in range(data.shape[0]):
            data_i = data[i]
            if zoom_factor is not None:
                data[i] = zoom(data[i], zoom_factor)
            fig = self._get_fig()
            plt.imshow(data[i], origin="lower", cmap='RdYlBu', vmin=vmin, vmax=vmax)
            rendered_list += [self._render(fig, expand=False)]

        return np.stack(rendered_list)


class TensorBoardScatterPlot(TensorBoardPlot):

    def plot(self, x, y, z, x_min, x_max, y_min, y_max, vmin=None, vmax=None, regression=False):

        assert len(x.shape) == 2 and len(y.shape) == 2 and len(z.shape) == 2

        rendered_list = []
        for i in range(x.shape[0]):
            fig = self._get_fig()

            plt.scatter(x[i], y[i], c=z[i], s=50, cmap='RdYlBu', vmin=vmin, vmax=vmax)
            plt.axis((x_min, x_max, y_min, y_max))

            rendered_list += [self._render(fig, expand=False)]

        return np.stack(rendered_list)


class TensorBoardRegressionPlot(TensorBoardPlot):

    def plot(self, x, y, x_min, x_max, y_min, y_max):

        assert len(x.shape) == 1 and len(y.shape) == 1

        fig = self._get_fig()

        fit = np.polyfit(x, y, deg=1)
        plt.plot(x, fit[0] * x + fit[1], color='red')
        plt.scatter(x, y, s=50)
        plt.axis((x_min, x_max, y_min, y_max))

        return self._render(fig, expand=True, axis=True)


class TensorBoardBarPlot(TensorBoardPlot):

    def plot(self, means, stds, group_labels, cond_labels, y_max=None, y_min=None, title=None):
        """Creates a bar plot from means and stds.

        Args:
            means: A [num_conditions x num_groups] array of means.
            std: A [num_conditions x num_groups] array of standard errors.
            group_labels: A list of str labels for the group (x-axis major index).
            cond_labels: A list of str labels for the group (x-axis minor index).

        Returns:
            TODO:
        """
        num_conds, num_groups = means.shape

        index = np.arange(num_groups)
        bar_width = 0.2
        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        fig = self._get_fig()

        for i, mean, std, lbl, color in zip(index, means, stds, cond_labels, colors):

            plt.bar(index + i * bar_width, mean, bar_width,
                    alpha=opacity,
                    color=color,
                    yerr=std,
                    error_kw=error_config,
                    label=lbl,
                    )

        if y_max is not None and y_min is not None:
            assert y_max is not None and y_min is not None
            plt.ylim([y_min, y_max])

        plt.xlabel('Number of Training Examples')
        plt.ylabel('Generalization Probability')
        plt.xticks(index + num_groups * bar_width / num_conds, group_labels)
        plt.legend()
        plt.tight_layout()
        if title is not None:
            plt.title(title)

        return self._render(fig, expand=True, axis=True)
