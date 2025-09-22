from matplotlib.path import Path
import numpy as np
from matplotlib.widgets import LassoSelector
from warnings import warn


class LassoSelect:
    def __init__(self, fig, ax, points, **selected_kwargs):
        self.fig = fig
        self.ax = ax
        self.points = points
        self.selected_points = self.ax.scatter([], [], **selected_kwargs)
        self.lasso = LassoSelector(ax, self.on_select)

    def on_select(self, verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.points))[0]
        self.selected_points.set_offsets(self.points[ind])
        self.fig.canvas.draw_idle()


class LassoMultipleSelect:
    def __init__(self, fig, ax, points, colors=None, **selected_kwargs):
        self.fig = fig
        self.ax = ax
        self.points = points
        self.lasso = LassoSelector(ax, self.on_select)
        self.data_list = []
        if colors is None:
            self.colors = [f'C{i}' for i in range(10)]
        else:
            self.colors = colors
        if 'c' in selected_kwargs or 'color' in selected_kwargs:
            raise ValueError('LassoMultipleSelected only accepts'
                             ' a color_list rather than a single color!')
        self.kwargs = selected_kwargs

    def on_select(self, verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.points))[0]
        self.data_list.append(self.points[ind])
        if len(self.data_list) == 1 + len(self.colors):
            warn(f'The number of colors ({len(self.colors)}) is not enough for selected groups!')
        color_ind = (len(self.data_list) - 1) % len(self.colors)
        self.ax.scatter(self.points[ind, 0], self.points[ind, 1], color=self.colors[color_ind], **self.kwargs)
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle

    # Generate random points
    np.random.seed(42)
    points = np.random.rand(1000, 2)

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=50, c='k', alpha=0.5)
    lasso = LassoSelect(fig, ax, points[points[:, 1] > 0.5], s=70, c='C3')
    # lasso = LassoMultipleSelect(fig, ax, points, colors=['C1', 'C2', 'C3'])
    # plt.show()
    # with open('tmp.pkl', 'wb') as f:
    #     pickle.dump(lasso.data_list, f)
    # with open('tmp.pkl', 'rb') as f:
    #     se = pickle.load(f)
    # for i, each in enumerate(se):
    #     plt.scatter(each[:, 0], each[:, 1], c=f'C{i+1}')
    plt.show()
