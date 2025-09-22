"""
Functions in this file only deal with
the interface variables, i.e., tension and strain
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
from utils.result_path import RES_PATH
import pickle

case_name = 'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
case_path = f'{RES_PATH}/{case_name}'
read_step = 1250000
Ncallback_lmp = 50
dt_lmp = 2e-5
st = np.load(f'{case_path}/interface/strain_tension_{read_step}.npz')
strain = st['strain']
tension = st['tension'] * 1e6
font_ticks = {'size': 17, 'family': 'Arial'}
font_label = {'size': 20, 'family': 'Arial'}
r_si = 2  # mm
n_rings = 200


def strain_map(ax=None, hlines=None):
    """
    :param ax: external, optional
    :param hlines: same-t line (ms)
    """
    from utils.id2x import x2ringID

    vmin, vmax, vsep = -0.41, 0.2, 0
    print(strain.shape, strain.min(), strain.max())
    cmax = np.array([70, 130, 180]) / 256
    cmin = np.array([139, 0, 0]) / 256
    from utils.post_process import customize_cmap
    external_ax = False
    if ax:
        external_ax = True
    else:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(figsize=(6, 5))
    ims = ax.imshow(
        strain, aspect='auto',
        cmap=customize_cmap(vmin, vmax, vsep, cmin, cmax, csep=np.array([1., 1., 1.])),
        vmin=vmin, vmax=vmax
    )
    cb = plt.colorbar(ims, location='bottom')
    # cb.set_ticks(ticks=np.arange(-0.4, 0.21, 0.2),
    #              labels=[f'{i:.1f}' for i in np.arange(-0.4, 0.21, 0.2)], fontdict=font_ticks)
    cb.set_label('Strain', fontdict=font_label,
                 # labelpad=-60
                 )
    ytick_labels = np.arange(0, 26, 5)
    yticks = ytick_labels / (Ncallback_lmp * dt_lmp) - 1
    xtick_labels = np.arange(10, 40, 10)
    xticks = x2ringID(xtick_labels)
    ax.set_ylim([yticks[-1], yticks[0]])
    if hlines:
        ax.hlines(np.array(hlines) * 1000 - 1, xmin=0, xmax=199, color='k', ls='--')
    if external_ax:
        return xticks, xtick_labels, yticks, ytick_labels
    plt.xticks(xticks, xtick_labels, fontdict=font_ticks)
    plt.yticks(yticks, ytick_labels, fontdict=font_ticks)
    plt.xlabel('Axial position x (mm)', fontdict=font_label)
    plt.ylabel('Time (sec)', fontdict=font_label)
    plt.tight_layout()
    fig.savefig(f'{case_path}/strainMap.png', dpi=300)
    plt.show()


def interpolate_ring_strain(points):
    """
    You cannot get strain at 0 ms with this method
    :param points: ndarray with shape of (N,2), 0th col is time and 1st col is x
    """
    from scipy.interpolate import RegularGridInterpolator as RGI
    strain_finer = RGI((np.arange(1, strain.shape[0] + 1),  # t (ms)
                        np.arange(strain.shape[1]),  # ringID
                        ), strain, method='linear')
    # ynew = np.linspace(0, 199, 1000)
    # xnew = np.linspace(1, 24999, 2000)
    # xnew, ynew = np.meshgrid(xnew, ynew)
    # plt.imshow(strain_finer((xnew, ynew)), aspect='auto')
    # plt.show()
    return strain_finer(points)


def interpolate_tension(points):
    """
    You cannot get tension at 25.0 ms with this method
    :param points: ndarray with shape of (N,2), 0th col is time and 1st col is x
    """
    from scipy.interpolate import RegularGridInterpolator as RGI
    tension_finer = RGI((np.arange(tension.shape[0]),  # t (ms)
                         np.arange(tension.shape[1]),  # ringID
                         ), tension, method='linear')
    # ynew = np.linspace(0, 199, 1000)
    # xnew = np.linspace(1, 24999, 2000)
    # xnew, ynew = np.meshgrid(xnew, ynew)
    # plt.imshow(strain_finer((xnew, ynew)), aspect='auto')
    # plt.show()
    return tension_finer(points)


def strain_single_ring(ringID, diameter=False, ax=None):
    if ax is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots()
        external_ax = False
    else:
        external_ax = True
    if diameter:
        ax.plot((1+strain[:, ringID]) * r_si * 2)
    else:
        ax.plot(strain[:, ringID])
    if not external_ax:
        ax.set_xlabel('Time (ms)', fontdict=font_label)
        ax.set_ylabel('Diameter (mm)' if diameter else 'Strain', fontdict=font_label)
        plt.show()


def wave_vel_advanced(ax=None):
    from scipy.signal import convolve2d
    from utils.mathfunc import slope_estimate_batch
    from functools import partial
    from utils.matplotlib_interact import LassoMultipleSelect

    dL = 2e-4 * 1e3  # mm
    dt = 1e-3
    sigma_t = 100 / 6
    n_time_half = int(sigma_t * 3)
    interval_arrow = 100
    scale = int(4000 / interval_arrow)
    quiver = partial(plt.quiver, scale=scale, angles='xy',
                     width=0.003, headwidth=3, headlength=5, headaxislength=4.5)
    print(strain.shape)
    t = np.arange(-n_time_half, n_time_half+1)
    G_t = -t / (np.sqrt(2 * np.pi) * sigma_t ** 3*dt) * np.exp(-t ** 2 / (2 * sigma_t ** 2))
    G_t = G_t[:, np.newaxis]
    strain_dt = convolve2d(strain, G_t, mode='same', boundary='symm')
    vmin, vmax = strain_dt.min(), strain_dt.max()
    print(vmin, vmax)
    strain_dtdt = convolve2d(strain_dt, G_t, mode='same', boundary='symm')
    contraction_line = np.where((np.abs(strain_dt) < 0.07) & (strain_dtdt > 0) & (strain < -0.07))
    contraction_line = np.array(contraction_line).T
    contraction_line[:, 0] = contraction_line[:, 0] + 1  # time index of 0 on the strain_map is actually 1 ms

    # LassoSelection ======================
    fig_i, ax_i = plt.subplots()
    ax_i.scatter(contraction_line[:, 0], contraction_line[:, 1], c='k')
    ax_i.invert_yaxis()
    lasso = LassoMultipleSelect(fig_i, ax_i, contraction_line)
    plt.show()
    with open(f'{case_path}/wave_labels.pkl', 'wb') as pf:
        pickle.dump(lasso.data_list, pf)
    # =====================================
    print(contraction_line.shape[0])
    with open(f'{case_path}/wave_labels.pkl', 'rb') as pf:
        groups = pickle.load(pf)

    if os.path.exists(f'{case_path}/slope_scale_tx_smooth_list.pkl'):
        with open(f'{case_path}/slope_scale_tx_smooth_list.pkl', 'rb') as pf2:
            slope_scale_tx_smooth_list = pickle.load(pf2)
    else:
        slope_scale_tx_smooth_list = slope_estimate_batch(
            groups, scale=dL / dt, remove_frac_end=0.,
            diff_dx=1, res_dx=4
        )
        with open(f'{case_path}/slope_scale_tx_smooth_list.pkl', 'wb') as pf2:
            pickle.dump(slope_scale_tx_smooth_list, pf2)
    slope_scale_tx_smooth = np.vstack(slope_scale_tx_smooth_list)
    vel = slope_scale_tx_smooth[:, 1]
    if ax is None:
        print(vel.min(), vel.max(), vel.mean(), np.median(vel))
        # t=tan(theta), u=cos(theta)=1/(sqrt(1+t^2)), v=sin(theta)=t/(sqrt(1+t^2))
        uv = np.c_[np.ones(slope_scale_tx_smooth.shape[0]), slope_scale_tx_smooth[:, 0]]
        uv /= (uv ** 2).sum(axis=1, keepdims=True) ** 0.5
        fig0, ax0 = plt.subplots(2, 1)
        for i, each in enumerate(groups):
            ax0[0].scatter(each[:, 0], each[:, 1], c=f'C{i}')
        ax0[0].invert_yaxis()
        cb = ax0[1].scatter(
            slope_scale_tx_smooth[:, 2], slope_scale_tx_smooth[:, 3], s=5, c=vel)
        ax0[1].invert_yaxis()
        plt.sca(ax0[1])
        quiver(slope_scale_tx_smooth[::interval_arrow, 2], slope_scale_tx_smooth[::interval_arrow, 3],
               uv[::interval_arrow, 0], uv[::interval_arrow, 1])
        plt.colorbar(cb)
        # plt.show()
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
        cb = ax[0].imshow(strain_dt, cmap='bwr', aspect='auto')
        ax[0].plot(contraction_line[:, 1], contraction_line[:, 0], 'g.', markersize=1)
        plt.colorbar(cb)
        cb = ax[1].imshow(strain, cmap='bwr', aspect='auto')
        plt.sca(ax[1])
        quiver(slope_scale_tx_smooth[::interval_arrow, 2], slope_scale_tx_smooth[::interval_arrow, 3],
               uv[::interval_arrow, 0], uv[::interval_arrow, 1])
        plt.colorbar(cb)
        cb = ax[2].imshow(strain_dtdt, cmap='bwr', aspect='auto')
        plt.colorbar(cb)

        fig3 = plt.figure(3)
        for each in slope_scale_tx_smooth_list:
            plt.plot(each[:, 3], each[:, 1], lw=2)
        plt.show()
    else:
        for each in slope_scale_tx_smooth_list:
            ax.plot(each[:, 2] / 1000, each[:, 1], lw=2)


def waveID_global2local():
    """
    Note that the global 5th wave can be the 4th wave of SMC #90.
    Useful in plot_strain_troughs and plot_tension_peaks.
    Users always specify the global waveIDs, but the program needs the local ones.
    """
    from math import floor, ceil
    print('Create file waveID_global2local.pkl ...')
    with open(f'{case_path}/slope_scale_tx_smooth_list.pkl', 'rb') as pf2:
        slope_scale_tx_smooth_list = pickle.load(pf2)

    global2local = [dict() for _ in range(n_rings)]
    for waveID, slope_scale_tx_smooth in enumerate(slope_scale_tx_smooth_list):
        x = slope_scale_tx_smooth[:, -1]
        ringID_min = floor(x.min())
        ringID_max = ceil(x.max())
        for ringID in range(ringID_min, ringID_max + 1):
            global2local[ringID][waveID] = len(global2local[ringID])
    with open(f'{case_path}/waveID_global2local.pkl', 'wb') as pf:
        pickle.dump(global2local, pf)
    print('Finish creating.')


def contraction_rate():
    """
    Define the peak point as (t_p, x_p). Extract all the points (t, x) subject to
        abs(t-t_p)<=0.5 sec=500 ms and x=x_p, also abs(strain_dt(x, t))>0.5 mm/s.
    :return:
    """
    from scipy.signal import convolve2d
    from scipy.interpolate import RegularGridInterpolator as RGI

    dt = 1e-3
    t_range = 1000  # ms
    n_time = 100
    sigma_t = n_time / 6
    t = np.arange(n_time) - np.arange(n_time).mean()
    G_t = -t / (np.sqrt(2 * np.pi) * sigma_t ** 3) * np.exp(-t ** 2 / (2 * sigma_t ** 2))
    G_t = G_t[:, np.newaxis]
    strain_dt = convolve2d(strain, G_t, mode='same', boundary='symm')
    im = plt.imshow(strain_dt / dt * r_si, aspect='auto', vmin=-2, vmax=2, cmap='bwr')
    plt.colorbar(im)
    plt.show()
    strain_dtdt = convolve2d(strain_dt, G_t, mode='same', boundary='symm')
    contraction_line = np.where((np.abs(strain_dt) < 7e-5) & (strain_dtdt > 0) & (strain < -0.07))
    contraction_line = np.array(contraction_line).T
    contraction_line[:, 0] = contraction_line[:, 0] + 1  # time index of 0 on the strain_map is actually 1 ms
    strain_dt_finer = RGI((np.arange(1, strain.shape[0] + 1),  # t (ms)
                           np.arange(strain.shape[1]),  # ringID
                           ), strain_dt / dt * r_si,  # mm/s
                          method='linear')
    with open(f'{case_path}/slope_scale_tx_smooth_list.pkl', 'rb') as pf2:
        slope_scale_tx_smooth_list = pickle.load(pf2)
    slope_scale_tx_smooth_all = np.vstack(slope_scale_tx_smooth_list)
    ts = np.linspace(-1000, 1000, 100) + slope_scale_tx_smooth_all[:, 2]
    points = np.c_[ts, slope_scale_tx_smooth_all[:, 3]]
    plt.scatter(points[:, 1], points[:, 0], s=1)
    # plt.plot(slope_scale_tx_smooth_all[:, 2], strain_dt_finer(slope_scale_tx_smooth_all[:, 2:]))
    # plt.plot(contraction_line[:, 0], strain_dt_finer(contraction_line))
    plt.show()


def extract_wall_move_vel(t, y, target_level=0.05, ringID=0):
    """
    Extract vels from target level to trough and from trough to target level
    Only include pairs where the wave rises above target_level after the trough

    :param t: time array
    :param y: amplitude array
    :param target_level: target level, default is 0.05
    :param ringID: help debug
    :return: Two lists of fall vels and rise vels
    """
    from scipy.interpolate import interp1d
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    from utils.mathfunc import root_nearest_larger, root_nearest_smaller
    # 1. Denoise
    y_orig = y.copy()
    y = gaussian_filter1d(y, 10)
    f_y = interp1d(t, y)

    # 2. Find all troughs
    # Find peaks of -y to get troughs of y, the strain difference>0.03, the strain<-0.2
    trough_ids, _ = find_peaks(-y, prominence=0.03, height=0.2)
    peak_ids, _ = find_peaks(y, prominence=0.05)
    # import matplotlib.pyplot as plt
    # plt.plot(t, y_orig)
    # plt.plot(t, y, '--')
    # plt.plot(t[trough_ids], y[trough_ids], 'C2.')
    # plt.show()

    # 3. Associate crossings with adjacent troughs and calculate vels
    fall_vels = []
    rise_vels = []
    trough_times = []

    # For each trough, find the nearest smaller and larger roots
    for i, trough_id in enumerate(trough_ids):
        # assert the smaller maximum > target level and then find the root
        nearest_smaller_peak_id = peak_ids[peak_ids < trough_id]
        if nearest_smaller_peak_id.size:
            smaller_maximum = y[nearest_smaller_peak_id[-1]]
        else:
            smaller_maximum = np.max(y[:trough_id])
        if smaller_maximum < target_level:
            continue
        trough_time = t[trough_id]
        down_cross_time = root_nearest_smaller(f_y, trough_time, step=0.1, max_range=2)
        if np.isnan(down_cross_time):
            print(f'ringID {ringID}, trough_time={trough_time} cannot find down_cross_time!')
            continue
        # assert the larger maximum > target level and then find the root
        nearest_larger_peak_id = peak_ids[peak_ids > trough_id]
        if nearest_larger_peak_id.size:
            larger_maximum = y[nearest_larger_peak_id[0]]
        else:
            larger_maximum = np.max(y[trough_id:])
        if larger_maximum < target_level:
            continue
        up_cross_time = root_nearest_larger(f_y, trough_time, step=0.1, max_range=2)
        if np.isnan(up_cross_time):
            print(f'ringID {ringID}, trough_time={trough_time} cannot find up_cross_time!')
            continue
        # Calculate vels
        fall_vel = abs(y[trough_id] * r_si * 2 / (trough_time - down_cross_time))
        rise_vel = abs(y[trough_id] * r_si * 2 / (up_cross_time - trough_time))

        # Add to result list
        fall_vels.append(fall_vel)
        rise_vels.append(rise_vel)
        trough_times.append(trough_time)
    return np.c_[fall_vels, rise_vels, trough_times]


def contraction_relaxation_vel(ax=None):
    from tqdm import tqdm

    if ax is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(layout='constrained')
        external_ax = False
    else:
        external_ax = True
    t = (np.arange(strain.shape[0]) + 1) / 1000
    res = []
    i = 8
    for each_ring in tqdm(strain.T[i:-8]):
        fall_rise_trough = extract_wall_move_vel(t, each_ring, target_level=0., ringID=i)
        if fall_rise_trough.shape[0]:
            res_cur = np.hstack((fall_rise_trough, np.full((fall_rise_trough.shape[0], 1), i)))
            res.append(res_cur)
            i += 1
    res = np.vstack(res)
    # import plotly.express as px
    # df = pd.DataFrame(res, columns=['fall', 'rise', 't', 'ringID'])
    # figure = px.scatter(df,x='fall',y='rise',hover_data=['t','ringID'])
    # figure.show()
    ax.scatter(res[:, 1], res[:, 0], alpha=0.5, s=5)
    vel_max = res[:, [0, 1]].max()
    ax.plot([0, vel_max], [0, vel_max], 'C1--')
    ax.plot([0, vel_max], [0, 3.5 * vel_max], 'C1--')
    if not external_ax:
        plt.axis('scaled')
        plt.xlim([0, vel_max + 0.1])
        plt.ylim([0, vel_max + 0.1])
        plt.xlabel('Relaxation rate (mm/s)', fontdict=font_label)
        plt.ylabel('Contraction rate (mm/s)', fontdict=font_label)
        fig.savefig(f'{case_path}/contraction_relaxation_rate.png', dpi=300)
        plt.show()


def extract_wave_front(peaks: np.ndarray, method='threshold', param=0.05) -> np.ndarray:
    """
    t is of unit ms, and
    x is actually ringID here according to the strain map
    :param peaks: the wave peaks, ndarray with shape of (N,2), 0th col is time and 1st col is x
    :param method: supported methods are 'time', 'x', 'left-max', and 'left-thresh'
        For 'left-thresh', the wave front has the same spatial position,
        but an earlier time point at which the value decreases to the threshold.
    :param param: parameters required by the method
    :return: wave fronts corresponding to points with shape of (N,2)
    """
    res = peaks.copy().astype(float)
    assert method in ['time', 'x', 'left-max', 'left-thresh']
    if method == 'time':
        res[:, 0] = res[:, 0] + param
        res[(res[:, 0] < 1) | (res[:, 0] > 25000), 0] = np.nan
        return res
    if method == 'x':
        res[:, 1] = res[:, 1] + param
        res[(res[:, 1] < 0) | (res[:, 1] > 199), 1] = np.nan
        return res
    if method == 'left-max':
        param = 0
    from scipy.interpolate import RegularGridInterpolator as RGI
    from utils.mathfunc import find_nearest_smaller_max, root_nearest_smaller
    strain_finer = RGI((np.arange(1, strain.shape[1] + 1),  # t (ms)
                        np.arange(strain.shape[0])),  # ringID
                       strain.T, method='linear')
    for i, (t, x) in enumerate(peaks):
        def strain_same_x(t_):
            return strain_finer(np.c_[t_, np.full_like(t_, x)]) - param

        if method == 'left-max':
            res[i, 0] = find_nearest_smaller_max(strain_same_x, t, lower_bound=max(1, t - 3000))
        else:
            res[i, 0] = root_nearest_smaller(strain_same_x, t, step=200, max_range=3000)
    return res


def wave_vel_paper(ax_map=None, ax_vel=None, need_front_same_wave=None, need_peak_same_x=None):
    from utils.mathfunc import slope_estimate_batch
    from utils.id2x import x2ringID
    from scipy.interpolate import PchipInterpolator

    dL = 2e-4 * 1e3  # mm, dL of SMC is twice that of IPAN
    dt = 1e-3
    interval_arrow = 200
    scale = int(4000 / interval_arrow)
    # ax_map and ax_vel must be given or not given at the same time
    assert not (bool(ax_map) ^ bool(ax_vel))
    if ax_map is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(
            1, 2, layout='constrained',
            figsize=(10, 5)
        )
        ax_map, ax_vel = ax
        external_ax = False
    else:
        external_ax = True
    with open(f'{case_path}/wave_labels.pkl', 'rb') as pf:
        groups = pickle.load(pf)
    if os.path.exists(f'{case_path}/slope_scale_tx_smooth_list.pkl'):
        with open(f'{case_path}/slope_scale_tx_smooth_list.pkl', 'rb') as pf2:
            slope_scale_tx_smooth_list = pickle.load(pf2)
    else:
        slope_scale_tx_smooth_list = slope_estimate_batch(
            groups, scale=dL / dt, remove_frac_end=0.,
            diff_dx=1, res_dx=4
        )
        with open(f'{case_path}/slope_scale_tx_smooth_list.pkl', 'wb') as pf2:
            pickle.dump(slope_scale_tx_smooth_list, pf2)
        waveID_global2local()
    cmap = plt.get_cmap('cividis')
    colors = cmap(np.linspace(0, 0.9, len(groups)))
    if need_front_same_wave is None:
        need_front_same_wave = []
    fronts_same_wave = []
    if need_peak_same_x is None:
        need_peak_same_x = []
    peaks_same_x = []
    for i, slope_scale_tx_smooth in enumerate(slope_scale_tx_smooth_list):
        if i in need_front_same_wave:
            front = extract_wave_front(slope_scale_tx_smooth[:, 2:], method='x', param=10)
            front = front[np.all(~np.isnan(front), axis=1)]
            fronts_same_wave.append(front)
            ax_map.plot(front[:, 1], front[:, 0], ls='--', c=colors[i])
        for x in need_peak_same_x:
            # the wave must cross this x-line so that the peak can be detected
            if slope_scale_tx_smooth[:, -1].min() < x < slope_scale_tx_smooth[:, -1].max():
                f_t = PchipInterpolator(slope_scale_tx_smooth[:, -1], slope_scale_tx_smooth[:, -2])
                peaks_same_x.append(f_t(x).item())
                if i == 0:
                    ax_map.vlines(x, ymin=0, ymax=peaks_same_x[-1], color=colors[i], ls='--')
                    # ax_map.plot([50,50],[0,peaks_same_x[-1]],'--', c=colors[i])
                    # plt.show()
                else:
                    ax_map.vlines(x, ymin=peaks_same_x[-2], ymax=peaks_same_x[-1], color=colors[i], ls='--')
        uv = np.c_[slope_scale_tx_smooth[:, 0], np.ones(slope_scale_tx_smooth.shape[0])]
        uv /= (uv ** 2).sum(axis=1, keepdims=True) ** 0.5
        # ax_map.quiver(
        #     slope_scale_tx_smooth[::interval_arrow, 3], slope_scale_tx_smooth[::interval_arrow, 2],
        #     uv[::interval_arrow, 0], uv[::interval_arrow, 1],
        #     scale=scale, angles='xy', color='k',
        #     width=0.006, headwidth=5, headlength=5, headaxislength=4.5,
        #     linewidth=10
        # ) # add an outline
        ax_map.quiver(
            slope_scale_tx_smooth[::interval_arrow, 3], slope_scale_tx_smooth[::interval_arrow, 2],
            uv[::interval_arrow, 0], uv[::interval_arrow, 1],
            scale=scale, angles='xy', color=colors[i],
            width=0.006, headwidth=5, headlength=5, headaxislength=4.5,
            linewidth=2
        )
        ax_vel.plot(slope_scale_tx_smooth[:, 3], slope_scale_tx_smooth[:, 1], lw=2, color=colors[i])
    ytick_labels = np.arange(0, 26, 5)
    yticks = ytick_labels / (Ncallback_lmp * dt_lmp) - 1
    xtick_labels = np.arange(10, 40, 10)
    xticks = x2ringID(xtick_labels)
    ax_map.set_ylim(yticks[-1], yticks[0])
    ax_map.set_xlim([0, 199])
    if external_ax:
        return xticks, xtick_labels, yticks, ytick_labels, fronts_same_wave, peaks_same_x
    ax_map.set_xticks(xticks)
    ax_map.set_xticklabels(xtick_labels)
    ax_map.set_yticks(yticks)
    ax_map.set_yticklabels(ytick_labels)
    ax_map.set_ylabel('Time (s)', fontdict=font_label)
    ax_map.set_xlabel('x (mm)', fontdict=font_label)
    ax_vel.set_ylabel('Wave velocity (mm/s)', fontdict=font_label)
    plt.show()



def plot_strain_troughs(waveID_global, ringIDs=None, ax=None, use_reverse_waveID=False):
    from scipy.signal import find_peaks
    from utils.id2x import x2ringID
    with open(f'{case_path}/waveID_global2local.pkl', 'rb') as pf:
        waveID_global2local = pickle.load(pf)
    if ax is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(layout='constrained')
        external_ax = False
    else:
        external_ax = True
    if ringIDs is None:
        ringIDs = np.arange(8, 200 - 8)
    troughs = np.zeros_like(ringIDs, dtype=float)
    for i, ringID in enumerate(ringIDs):
        # ringID = 148
        trough_ids, _ = find_peaks(-strain[:, ringID], prominence=0.03, height=0.1)
        # t = np.arange(1,25001)
        # plt.plot(t, strain[:, ringID])
        # plt.scatter(t[trough_ids], strain[trough_ids, ringID], s=60)
        # plt.show()
        if waveID_global not in waveID_global2local[ringID]:
            print(f'ring #{ringID} is not involved in this wave')
            troughs[i] = np.nan
            continue
        waveID_local = waveID_global2local[ringID][waveID_global]
        if use_reverse_waveID:
            waveID_local_reverse = -len(waveID_global2local[ringID]) + waveID_local
            print(f'ring #{ringID}, reverseID is {waveID_local_reverse}')
            troughs[i] = strain[trough_ids[waveID_local_reverse], ringID]
        else:
            if len(trough_ids) <= waveID_local:
                print(f'ring #{ringID} has no such troughs')
                troughs[i] = np.nan
                continue
            troughs[i] = strain[trough_ids[waveID_local], ringID]
    ax.plot(ringIDs, troughs)
    if not external_ax:
        plt.show()
    # xtick_labels = np.arange(10, 40, 10)
    # xticks = x2ringID(xtick_labels)
    # plt.xticks(xticks, xtick_labels)


def plot_tension_peaks(waveID_global, ringIDs=None, ax=None, use_reverse_waveID=False):
    from scipy.signal import find_peaks
    from utils.id2x import x2ringID
    with open(f'{case_path}/waveID_global2local.pkl', 'rb') as pf:
        waveID_global2local = pickle.load(pf)
    if ax is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(layout='constrained')
        external_ax = False
    else:
        external_ax = True
    if ringIDs is None:
        ringIDs = np.arange(8, 200 - 8)
    peaks = np.zeros_like(ringIDs, dtype=float)
    for i, ringID in enumerate(ringIDs):
        SIPID = ringID - 8
        # ringID = 148
        peak_ids, _ = find_peaks(tension[:, SIPID], height=2)
        # t = np.arange(1,25001)
        # plt.plot(t, tension[:, SIPID])
        # plt.scatter(t[peak_ids], tension[peak_ids, SIPID], s=60)
        # plt.show()
        # t = np.arange(1,25001)
        # plt.plot(t, tension[:, ringID])
        # plt.scatter(t[peak_ids], tension[peak_ids, ringID], s=60)
        # plt.show()
        if waveID_global not in waveID_global2local[ringID]:
            print(f'ring #{ringID} is not involved in this wave')
            peaks[i] = np.nan
            continue
        waveID_local = waveID_global2local[ringID][waveID_global]
        if use_reverse_waveID:
            waveID_local_reverse = -len(waveID_global2local[ringID]) + waveID_local
            print(f'ring #{ringID}, reverseID is {waveID_local_reverse}')
            peaks[i] = tension[peak_ids[waveID_local_reverse], SIPID]
        else:
            if len(peak_ids) <= waveID_local:
                print(f'ring #{ringID} has no such peaks')
                peaks[i] = np.nan
                continue
            peaks[i] = tension[peak_ids[waveID_local], SIPID]
    ax.plot(ringIDs, peaks)
    if not external_ax:
        plt.show()
    # xtick_labels = np.arange(10, 40, 10)
    # xticks = x2ringID(xtick_labels)
    # plt.xticks(xticks, xtick_labels)


def plot_tension_peaks_bkp(waveID, SIPIDs=None, ax=None):
    from scipy.signal import find_peaks
    from utils.id2x import x2ringID
    if ax is None:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(layout='constrained')
        external_ax = False
    else:
        external_ax = True
    if SIPIDs is None:
        SIPIDs = np.arange(0, 200 - 8 * 2)
    peaks = np.zeros_like(SIPIDs, dtype=float)
    for i, SIPID in enumerate(SIPIDs):
        peak_ids, _ = find_peaks(tension[:, SIPID], prominence=0.15, height=0.1)
        # t = np.arange(1,25001)
        # plt.plot(t, strain[:, ringID])
        # plt.scatter(t[trough_ids], strain[trough_ids, ringID], s=60)
        # plt.show()
        if len(peak_ids) <= waveID:
            print(f'SMC #{SIPID} has no such peaks')
            continue
        peaks[i] = tension[peak_ids[waveID], SIPID]
    plt.plot(SIPIDs, peaks)
    if not external_ax:
        plt.show()
    # xtick_labels = np.arange(10, 40, 10)
    # xticks = x2ringID(xtick_labels)
    # plt.xticks(xticks, xtick_labels)


def tension_map(ax=None, vline=None):
    """
    :param ax: external, optional
    :param vline: same-t line (ms)
    """
    from utils.id2x import x2ringID
    vmin, vmax, vsep = 0, 15, 0
    tension_ = np.zeros((25000, 200))
    tension_[:, 8:-8] = tension
    print(tension_.shape, tension_.min(), tension_.max())
    if ax:
        external_ax = True
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        external_ax = False
    ims = ax.imshow(
        tension_, aspect='auto',
        cmap='viridis',
        vmin=vmin, vmax=vmax
    )
    cb = plt.colorbar(ims, location='right')
    cb_ticks = np.arange(0, 16, 5)
    cb.set_ticks(ticks=cb_ticks, labels=cb_ticks, fontdict=font_ticks)
    cb.set_label('Active force (μN)', fontdict=font_label)
    ytick_labels = np.arange(0, 26, 5)
    yticks = ytick_labels / (Ncallback_lmp * dt_lmp)
    xtick_labels = np.arange(10, 40, 10)
    xticks = x2ringID(xtick_labels)
    if vline:
        ax.axvline(vline, color='w', ls='--')
    if external_ax:
        return xticks, xtick_labels, yticks, ytick_labels
    plt.xticks(xticks, xtick_labels, fontdict=font_ticks)
    plt.yticks(yticks, ytick_labels, fontdict=font_ticks)
    plt.xlabel('x (mm)', fontdict=font_label)
    plt.ylabel('Time (s)', fontdict=font_label)
    plt.tight_layout()
    fig.savefig(f'{case_path}/tensionMap.png', dpi=300, transparent=True)
    plt.show()


def plot_strain_multiframe(frames, ringIDmin, ringIDmax, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        external_ax = False
    else:
        external_ax = True
    cmap = plt.get_cmap('cividis')
    colors = cmap(np.linspace(0, 0.9, len(frames)))
    for i, frame in enumerate(frames):
        ax.plot(strain[frame, ringIDmin:ringIDmax], color=colors[i], marker='v')
    if not external_ax:
        plt.show()


def plot_tension_multiframe(frames, ringIDmin, ringIDmax, ax=None):
    # im = plt.imshow(tension[frames[0]:frames[-1], ringIDmin:ringIDmax],aspect='auto')
    # plt.colorbar(im)
    # plt.show()
    if ax is None:
        fig, ax = plt.subplots()
        external_ax = False
    else:
        external_ax = True
    SIPIDmin, SIPIDmax = ringIDmin - 8, ringIDmax - 8
    cmap = plt.get_cmap('cividis')
    colors = cmap(np.linspace(0, 0.9, len(frames)))
    for i, frame in enumerate(frames):
        ax.plot(tension[frame, SIPIDmin:SIPIDmax], color=colors[i], marker='v')
    if not external_ax:
        plt.show()


def draft():
    from utils.mathfunc import detect_narrow_peaks_2d
    y = tension[:, [9100, 9200]]
    x = np.arange(y.shape[0])

    peaks, flag_narrow = detect_narrow_peaks_2d(
        y, min_height=5, min_peak_prominence=3, rest_value=0.2, min_peak_interval=10)
    # plt.plot(x, y, '.-', color='gray', alpha=0.3, label='Original')
    # plt.plot(x[flag_narrow], y[flag_narrow])
    fig, ax = plt.subplots(2, 1, sharex=True)
    for i in range(2):
        ax[i].plot(x, y[:, i], '.-', color='gray', alpha=0.3, label='Original')
        ax[i].plot(x[flag_narrow[:, i]], y[flag_narrow[:, i], i], '.')
        ax[i].scatter(x[peaks[i]], y[peaks[i], i], 50, marker='x', c='C1')
    plt.show()


if __name__ == '__main__':
    # ax = plt.axes()
    # strain_map()
    # plt.show()
    # interpolate_ring_strain(np.array([[1, 3]]))
    # strain_single_ring([8])
    # fig, ax = plt.subplots()
    wave_vel_advanced()
    # waveID_global2local()
    # contraction_rate()
    # contration_relaxation_vel()
    # extract_wave_front(np.array([[10000, 2], [20000, 50]]))
    # wave_vel_paper_old()
    # wave_vel_paper()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Wave velocity (mm/s)')
    # plt.show()

    # tension_map()
    # fig, ax = plt.subplots(2, 1, sharex=True, layout='constrained')
    # plot_strain_troughs(waveID_global=5, use_reverse_waveID=True)
    # plot_tension_peaks(waveID_global=5, use_reverse_waveID=True)
    # plot_peak_tension(need_peak_same_wave=[5], ax=ax[1])
    # plt.show()
    # plot_strain_multiframe(frames=list(range(17400-1, 18000, 100)), ringIDmin=25, ringIDmax=37)
    # plot_tension_multiframe(frames=list(range(17400 - 1, 18000, 100)), ringIDmin=25, ringIDmax=37)
    # draft()
