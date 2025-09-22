from utils.geometry import *
import numpy as np
from warnings import warn


class Geometry:
    def __init__(self, print_log=True):
        self.xs = np.array([])
        self.ys = np.array([])
        self.zs = np.array([])
        self.log = ''
        self.l_axis = 0
        self.n_axis = 0
        self.n_ring = 0
        self.print_log = print_log

    @property
    def size(self):
        return self.xs.size

    @property
    def flatten_coords(self):
        return np.c_[self.xs, self.ys, self.zs].flatten()

    def set_coord(self, xs, ys, zs):
        if isinstance(xs, int):
            self.ys = ys
            self.zs = zs
            self.xs = np.full_like(ys, xs)
        elif isinstance(ys, int):
            self.xs = xs
            self.zs = zs
            self.ys = np.full_like(xs, ys)
        elif isinstance(zs, int):
            self.xs = xs
            self.ys = ys
            self.zs = np.full_like(xs, zs)
        else:
            self.xs = xs
            self.ys = ys
            self.zs = zs
        return self

    def shift(self, x=0, y=0, z=0):
        new_geo = self._copy()
        new_geo.xs += x
        new_geo.ys += y
        new_geo.zs += z
        return new_geo

    def mirror(self, plane_name, plane_pos):
        new_geo = self._copy()
        if plane_name == 'YOZ':
            new_geo.xs = plane_pos * 2 - self.xs
        elif plane_name == 'XOY':
            new_geo.zs = plane_pos * 2 - self.zs
        elif plane_name == 'XOZ':
            new_geo.ys = plane_pos * 2 - self.ys
        else:
            raise ValueError('Invalid plane_name!')
        return new_geo

    def _copy(self):
        from copy import deepcopy
        new_geo = self.__class__.__new__(self.__class__)
        new_geo.__dict__.update(deepcopy(self.__dict__))
        return new_geo


class ThickRectangle(Geometry):
    def __init__(self, length, width, n_thick, plane_pos, axis, dl):
        super(ThickRectangle, self).__init__()
        z_bot = np.arange(0, length + dl * 0.1, dl).round(6)
        self.length = z_bot[-1]
        if self.length != length:
            warn('The length has been modified! Be careful to shift.')
            print(f'Real length is {self.length}')
        x_left = np.arange(dl, width + dl * 0.1, dl).round(6)
        self.width = x_left[-1]
        if self.width != width:
            warn('The width has been modified! Be careful to shift.')
            print(f'Real width is {self.width}')
        x_left = x_left[:-1]
        z_left = np.full_like(x_left, 0)
        x_right = np.copy(x_left)
        z_right = np.full_like(x_right, self.length)
        x_bot = np.full_like(z_bot, 0)
        z_top = np.copy(z_bot)
        x_top = np.full_like(z_top, self.width)
        if n_thick == 1:
            zs = np.r_[z_bot, z_left, z_top, z_right]
            xs = np.r_[x_bot, x_left, x_top, x_right]
            ys = np.full_like(xs, plane_pos)
        else:
            d = dict()
            xs, ys, zs = [], [], []
            for pos, axis, direction in zip(
                    ['bot', 'left', 'top', 'right'], ['x', 'z', 'x', 'z'], [-1, -1, 1, 1]):
                d[pos] = eval(f'Geometry().set_coord(x_{pos}, plane_pos, z_{pos})')
                d[pos] = Stack(d[pos], axis, direction * n_thick, dl)
                xs.append(d[pos].xs)
                ys.append(d[pos].ys)
                zs.append(d[pos].zs)
            xs = np.hstack(xs)
            ys = np.hstack(ys)
            zs = np.hstack(zs)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=axis)


class Rectangle(Geometry):
    def __init__(self, length, width, plane_pos, axis, dl):
        super().__init__()
        z = np.arange(0, length + dl * 0.1, dl).round(6)
        self.length = z[-1]
        if self.length != length:
            warn('The length has been modified! Be careful to shift.')
            print(f'Real length is {self.length}')
        x = np.arange(0, width + dl * 0.1, dl).round(6)
        self.width = x[-1]
        if self.width != width:
            warn('The width has been modified! Be careful to shift.')
            print(f'Real width is {self.width}')
        zs, xs = np.meshgrid(z, x)
        zs, xs = zs.flatten(), xs.flatten()
        ys = np.full_like(xs, plane_pos)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=axis)


class CylinderSide(Geometry):
    def __init__(
            self, r, l_axis, dl, axis='y',
            print_log=False
    ):
        super().__init__(print_log)
        self.l_axis = l_axis
        self.r = r

        self.n_axis = int(self.l_axis / dl) + 1
        # use y-axis as the cylinder-axis
        y = np.arange(0, self.n_axis) * dl
        self.l_axis = y[-1]
        self.log = f'Real length is {self.l_axis}.'
        if self.l_axis != l_axis:
            warn('The length has been modified! Be careful to shift.')
        if self.print_log:
            print(self.log)
        self.n_ring = get_n_per_ring(r, dl)
        z, x = rad2cart(self.n_ring, r)
        zs = np.tile(z, self.n_axis)
        xs = np.tile(x, self.n_axis)
        ys = np.repeat(y, self.n_ring)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=axis)

    @property
    def dl_in_ring(self):
        return get_dist_on_ring(self.r, self.n_ring)

    def get_and_delete(self, ind: int, direction: str):
        """
        Get layers of atom coordinate and delete these atoms.
        :param ind: int, e.g., {0}, [n_axis_si-1, n_axis_si].
        :param direction: str, 'smaller' will get & delete [0,ind),
        'bigger' will get & delete (ind, n_axis_si].
        :return: xs, ys, zs
        """
        coords = np.c_[self.xs, self.ys, self.zs].reshape((-1, self.n_ring, 3))
        ind_all = list(range(self.n_axis))
        if direction == 'smaller':
            ind_req = ind_all[:ind]
            ind_remain = ind_all[ind:]
        elif direction == 'larger':
            ind_req = ind_all[ind + 1:]
            ind_remain = ind_all[:ind + 1]
        else:
            raise ValueError('Direction can only be smaller or larger')
        x_req = coords[ind_req, :, 0].flatten()
        y_req = coords[ind_req, :, 1].flatten()
        z_req = coords[ind_req, :, 2].flatten()
        self.xs = coords[ind_remain, :, 0].flatten()
        self.ys = coords[ind_remain, :, 1].flatten()
        self.zs = coords[ind_remain, :, 2].flatten()
        self.n_axis = len(ind_remain)
        return x_req, y_req, z_req


class Torus(Geometry):
    def __init__(
            self, r_ring, r_t, dl, n_ring,
            plane='YOZ',
            phi_range='[180,270)', smallest_ID=None
    ):
        super().__init__()
        assert r_ring < r_t
        phi_min, phi_max = (float(each) for each in phi_range[1:-1].split(','))
        phi_tot = phi_max - phi_min
        assert phi_tot <= 360
        if phi_tot == 360:
            incl_min, incl_max = True, False
        else:
            if phi_range[0] == '[':
                incl_min = True
            elif phi_range[0] == '(':
                incl_min = False
            else:
                raise ValueError('Include sign can only be [ or (')
            if phi_range[-1] == ']':
                incl_max = True
            elif phi_range[-1] == ')':
                incl_max = False
            else:
                raise ValueError('Include sign can only be [ or (')
        self.atom_id_boundary = None

        thetas = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        r_P = r_t - r_ring * np.cos(thetas)
        phi_min = np.deg2rad(phi_min)
        phi_max = np.deg2rad(phi_max)
        phi_tot = phi_max - phi_min
        nLs = get_n_per_ring(r_P, dl, phi_ring=phi_tot)
        d_phis = phi_tot / nLs
        all_phi, all_theta = [], []
        for theta, nL, d_phi in zip(thetas, nLs, d_phis):
            cur_phis = d_phi * np.arange(int(not incl_min), nL + int(incl_max)) + phi_min
            all_phi.append(cur_phis)
            cur_thetas = theta * np.ones_like(cur_phis)
            all_theta.append(cur_thetas)
        self.theta_at_phi_min = 0
        self.theta_at_phi_max = 0
        if incl_min:
            self.theta_at_phi_min = all_theta[0].min()
        all_theta = np.hstack(all_theta)
        all_phi = np.hstack(all_phi)
        if smallest_ID:
            from pandas import DataFrame
            df = DataFrame({'theta': all_theta, 'phi': all_phi})
            df = df.sort_values(by='phi').reset_index()[:n_ring]
            df['index'] += smallest_ID
            df.sort_values(by='theta', inplace=True)
            self.atom_id_boundary = df['index'].to_list()
        xs = r_ring * np.sin(all_theta)
        ys = (r_t - r_ring * np.cos(all_theta)) * np.sin(all_phi)
        zs = (r_t - r_ring * np.cos(all_theta)) * np.cos(all_phi)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, plane=plane)

    def extend(self, xs, ys, zs, pos='end'):
        assert xs.size == ys.size == zs.size
        if pos == 'end':
            self.xs = np.r_[self.xs, xs]
            self.ys = np.r_[self.ys, ys]
            self.zs = np.r_[self.zs, zs]
        elif pos == 'start':
            self.xs = np.r_[xs, self.xs]
            self.ys = np.r_[ys, self.ys]
            self.zs = np.r_[zs, self.zs]
        else:
            raise ValueError('pos can only be end or start')


class ThickRing(Geometry):
    """
    A ring with thickness r_out-r_in, i.e.,
    the region between two concentric circles with radii of r_out and r_in.
    If r_in = 0, it is a filled circle.
    The thick rings can be stacked to create a region between two coaxial cylinders.
    """

    def __init__(
            self, r_out, r_in, dl, incl_inner, incl_outer, axis='y',
            adjust_dl=False, equal_size_per_circle=False
    ):
        super().__init__()
        self.dl = dl
        self.n_ring_in = get_n_per_ring(r_in, self.dl)
        if adjust_dl:
            assert r_in > 0
            self.dl = get_dist_on_ring(r_in, self.n_ring_in)
        n_radial = round((r_out - r_in) / self.dl)
        self.r_out = n_radial * self.dl + r_in
        self.log = f'Real r_out {self.r_out}, real dl {self.dl}'
        print(self.log)
        rs = np.arange(0, n_radial + 1) * self.dl + r_in
        if equal_size_per_circle:
            n_per_rings = np.full_like(rs, self.n_ring_in).astype(int)
        else:
            n_per_rings = get_n_per_ring(rs, self.dl)
            assert self.n_ring_in == n_per_rings[0]
        if not incl_inner:
            rs = rs[1:]
            n_per_rings = n_per_rings[1:]
        if not incl_outer:
            rs = rs[:-1]
            n_per_rings = n_per_rings[:-1]
        self.rs = rs
        self.n_per_rings = n_per_rings
        zs, xs = [], []
        for r, n in zip(self.rs, self.n_per_rings):
            z, x = rad2cart(n, r)
            zs.extend(z)
            xs.extend(x)
        zs, xs = np.asarray(zs), np.asarray(xs)
        ys = np.full_like(zs, 0)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=axis)


class Union(Geometry):
    def __init__(self, *bodies):
        super().__init__()
        self.xs = np.hstack(tuple(body.xs for body in bodies))
        self.ys = np.hstack(tuple(body.ys for body in bodies))
        self.zs = np.hstack(tuple(body.zs for body in bodies))


class Stack(Geometry):
    def __init__(self, plane, axis, n_axis, dl):
        # if n_axis_si>0, along the positive axis; else if n_axis_si<0, along the negative axis
        super().__init__()
        coord_plane = np.c_[plane.xs, plane.ys, plane.zs]
        axis2num = {'x': 0, 'y': 1, 'z': 2}
        level = coord_plane[0, axis2num[axis]]
        assert np.all(coord_plane[:, axis2num[axis]] == level)
        coords = np.zeros((np.abs(n_axis) * plane.xs.size, 3))
        for cur_axis, col in axis2num.items():
            if cur_axis == axis:
                coords[:, col] = np.repeat(np.arange(0, n_axis, np.sign(n_axis)) * dl, plane.xs.size) + level
            else:
                coords[:, col] = np.tile(coord_plane[:, col], np.abs(n_axis))
        self.xs = coords[:, 0]
        self.ys = coords[:, 1]
        self.zs = coords[:, 2]
