import numpy as np


def get_n_per_ring(r, d, phi_ring=2*np.pi):
    """"""
    n = np.ones_like(r).flatten()
    r = np.asarray(r).flatten()
    n[r != 0] = phi_ring / np.arccos(1 - 0.5 * (d / r[r != 0]) ** 2)
    if n.size == 1:
        return round(n.item())
    return np.round(n).astype(int)


def get_dist_on_ring(r, n):
    d = r * np.sqrt(2 * (1 - np.cos(2 * np.pi / n)))
    return d


def rad2cart(n, r):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x1, x2 = r * np.cos(angles), r * np.sin(angles)
    return x1, x2


def transform_coordinate(xs, ys, zs, **kwargs):
    """
    default coordinate system is x (inward) y (up) z (right)
    y
    | x
    |/
    ----z
    :param xs: self-explanatory
    :param ys: self-explanatory
    :param zs: self-explanatory
    :param kwargs: axis or plane, as described in plane2axis
    :return: xs, ys, zs in the new coordinate system
    """
    axis = 'y'
    plane2axis_up = {'XOY': 'x', 'YOZ': 'y', 'XOZ': 'z'}
    assert len(kwargs) == 1
    if 'axis' in kwargs:
        axis_up = kwargs['axis']
        assert axis_up in ['x', 'y', 'z']
    elif 'plane' in kwargs:
        plane = kwargs['plane']
        assert plane in plane2axis_up.keys()
        axis_up = plane2axis_up[plane]
    else:
        raise KeyError('Kwargs can only be axis or plane!')
    if axis_up == 'z':
        xs, ys, zs = zs, xs, ys
    elif axis_up == 'x':
        xs, ys, zs = ys, zs, xs
    return xs, ys, zs


def get_wall_ID(i, j, n_per_ring, smallest_ID=1):
    """
    get the ID of cylinder wall
    :param i: ID on the ring
    :param j: ID on the axis
    :param n_per_ring: self-explanatory
    :param smallest_ID: self-explanatory
    :return: wall-ID
    """
    if i > n_per_ring:
        i = i % n_per_ring
    return (j - 1) * n_per_ring + i + smallest_ID - 1


if __name__ == '__main__':
    r = 0.002
    n = get_n_per_ring(r, 5e-4)
    dl = get_dist_on_ring(r, n)
    x, z = rad2cart(n, r)
    dist = ((x[1:] - x[:-1]) ** 2 + (z[1:] - z[:-1]) ** 2) ** 0.5
    pass
