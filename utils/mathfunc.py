import numpy as np
from warnings import warn


def proj(a: np.ndarray, b: np.ndarray) -> float:
    # project vector a onto vector b
    # return the magnitude of the resultant vector (projection length)
    assert a.shape == b.shape
    if b.ndim == 1:
        res = np.dot(a, b) / np.sqrt(np.dot(b, b))
    else:
        res = np.sum(a * b, axis=-1) / np.sqrt(np.sum(b * b, axis=-1))
    return res


def slope_estimate(x, y, scale, remove_end_frac=0.,
                   diff_dx=0.5, res_dx=2, orig_frac=0.5, deri_frac=0.2):
    """
    :param x: x
    :param y: y with the same shape as x
    :param scale: incorporate the unit scaling
    :param remove_end_frac: the boundary often has different slope with the main part, remove it
    :param diff_dx: dx to calculate the difference
    :param res_dx: dx of the result
    :param orig_frac: smoothing factor of the original data
    :param deri_frac: smoothing factor of the derivative data
    :return: [slope, slope_sclaed, x_smooth, y_smooth]
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    res_dx_multiples = int(res_dx / diff_dx)
    if res_dx_multiples * diff_dx != res_dx:
        warn('res_dx had better be the multiple of diff_dx')
    # LOWESS smoothing
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    x_start = x_min + x_range * remove_end_frac
    x_end = x_max - x_range * remove_end_frac
    xs = np.arange(x_start, x_end + diff_dx, diff_dx)
    if orig_frac > 0:
        ys = lowess(y, x, frac=orig_frac, it=3, return_sorted=False, xvals=xs)
    else:
        ys = y

    # get derivative by central difference
    slopes = np.zeros_like(ys)
    slopes[1:-1] = (ys[2:] - ys[:-2]) / diff_dx * 0.5
    slopes[0] = (ys[1] - ys[0]) / diff_dx
    slopes[-1] = (ys[-1] - ys[-2]) / diff_dx
    if deri_frac > 0:
        slopes = lowess(slopes, xs, frac=deri_frac, it=2, return_sorted=False)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 1, sharex=True)
    # ax[0].scatter(x, y)
    # ax[0].plot(xs, ys, 'k.-')
    # ax[1].plot(xs, slopes,'.-')
    # plt.show()

    slopes_scale = slopes * scale

    return (slopes[::res_dx_multiples], slopes_scale[::res_dx_multiples],
            xs[::res_dx_multiples], ys[::res_dx_multiples])


def slope_estimate_batch(groups_xy, scale=1, remove_frac_end=0,
                         diff_dx=0.5, res_dx=2, orig_frac=0.5, deri_frac=0.2):
    """
    :param diff_dx: the interpolation interval in slope_estimate for differentiation
    :param res_dx: the interval of the result
    :param groups_xy: list of ndarray[x,y], [shape(10,2),shape(20,2),shape(4,2),...]
    :return: groups of ndarray[slope,slope_scale,x,y]
    """
    res = []
    for group in groups_xy:
        slope, slope_scale, x_smooth, y_smooth = slope_estimate(
            group[:, 0], group[:, 1], scale=scale, remove_end_frac=remove_frac_end,
            diff_dx=diff_dx, res_dx=res_dx,
            orig_frac=orig_frac, deri_frac=deri_frac)
        res.append(np.c_[slope, slope_scale, x_smooth, y_smooth])
    return res


def time_integrate(t: np.ndarray, d: np.ndarray, dt_scale=1000):
    from scipy.interpolate import PchipInterpolator
    from scipy.integrate import cumulative_simpson

    # t must be equally spaced
    assert np.allclose(np.diff(np.diff(t)), 0, rtol=1e-10, atol=1e-10)
    if dt_scale < 1:
        raise ValueError('dt_scale must be >= 1')
    if d.ndim == 1:
        d = d[np.newaxis, :]
        flag_1d = True
    else:
        flag_1d = False
    dt = (t[1] - t[0]) / dt_scale
    interp = PchipInterpolator(t, d, axis=1)
    ts = np.arange(t[0], t[-1] + dt, dt)
    ds = interp(ts)
    fs = cumulative_simpson(ds, x=ts, initial=0)
    f = fs[:, 0::dt_scale]
    if flag_1d:
        f = f[0]
    return f


def find_clusters(points, threshold, min_population=0):
    """
    Find clusters with its start indices and end indices.
    Neighboring points with distance<threshold belong to a same group.

    :param points: ascending sorted 1d array
    :param threshold: distance threshold
    :param min_population: the minimal number of points the cluster should have so that the cluster won't be too short

    :returns: Two 1d arrays, one is start_indices and the other end_indices
    """
    points = np.asarray(points)
    n = len(points)
    if n < 2:
        return [], []
    # import matplotlib.pyplot as plt
    # plt.eventplot(points)
    # plt.show(block=True)

    diffs = np.diff(points)
    # Determine where should be broken by the threshold
    break_indices = np.where(diffs >= threshold)[0] + 1

    start_indices = np.r_[[0], break_indices]
    end_indices = np.r_[break_indices, [n]]
    populations = end_indices - start_indices
    # print(populations)
    mask = populations > min_population
    # start_values = []
    # end_values = []
    # for start_idx, end_idx in zip(start_indices, end_indices):
    #     if end_idx - start_idx > 0:
    #         start_values.append(points[start_idx])
    #         end_values.append(points[end_idx] + 1)
    return start_indices[mask], end_indices[mask]


def detect_narrow_peaks(y, min_peak_prominence=0.1, min_height=5,
                        rest_value=0.1, max_width=10,
                        min_peak_interval=3):
    """
    Detect narrow peaks. Use it in visual_sph.compare_Fa_strain to
    fiter the narrow peaks which can have smaller strain but large active force.
    :param y: data, 1d array
    :param min_peak_prominence: the difference between the peak and the ground. Use it to filter the small peaks.
    :param min_height: the absolute height of the peak. Use it to filter the low peaks.
    :param rest_value: resting value to determine the region bounds
    :param max_width: peaks lower than this width is narrow
    :param min_peak_interval:
    :return: narrow_mask, 1d array
    """
    from scipy.signal import find_peaks
    # Detect narrow peaks
    peaks, _ = find_peaks(
        y,
        prominence=min_peak_prominence,
        height=min_height,
        width=(0, max_width)
    )
    start_indices, end_indices = find_clusters(peaks, min_peak_interval)
    # Create narrow mask
    narrow_mask = np.zeros_like(y, dtype=bool)
    for start_idx, end_idx in zip(start_indices, end_indices):
        left_base = np.argwhere(y[:start_idx] < rest_value)
        left_base = left_base.max() if left_base.size else 0
        right_base = np.argwhere(y[end_idx:] < rest_value)
        right_base = right_base.min() + end_idx + 1 if right_base.size else narrow_mask.size - 1
        narrow_mask[left_base:right_base + 1] = True
    return peaks, narrow_mask


def detect_narrow_peaks_2d(y, axis=0, **kwargs):
    from tqdm import tqdm
    peaks = []
    narrow_masks = np.zeros_like(y, dtype=bool)
    if axis == 1:
        for i, signal in enumerate(tqdm(y)):
            peak, narrow_masks[i] = detect_narrow_peaks(signal, **kwargs)
            peaks.append(peak)
    elif axis == 0:
        for i, signal in enumerate(tqdm(y.T)):
            peak, narrow_masks[:, i] = detect_narrow_peaks(signal, **kwargs)
            peaks.append(peak)
    else:
        raise KeyError('Axis can only be 0 or 1.')
    return peaks, narrow_masks


def root_nearest_smaller(f, b, step=0.1, max_range=10, a_min=None) -> float:
    """
    Find the nearest smaller root
    :param f: function
    :param b: upper bound
    :param step: self-explanatory
    :param max_range: maximal range of search
    :param a_min: allowed minimum b
    :return:
    """
    # return b if it is the root per se
    if abs(f(b)) < 1e-10:
        return b
    if a_min is None:
        a_min = np.min(f.x)
    from scipy.optimize import brentq
    max_iter = int(max_range / step)
    sign_b = np.sign(f(b))
    for i in range(1, max_iter + 1):
        a_now = max(b - i * step, a_min)
        if np.sign(f(a_now)) == sign_b:
            continue
        root = brentq(f, a_now, b)
        return root
    return np.nan


def root_nearest_larger(f, a, step=0.1, max_range=10, b_max=None) -> float:
    """
    Find the nearest larger root
    :param f: function
    :param a: lower bound
    :param step: self-explanatory
    :param max_range: maximal range of search
    :param b_max: allowed maximal b
    :return:
    """
    # return a if it is the root per se
    if abs(f(a)) < 1e-10:
        return a
    from scipy.optimize import brentq
    if b_max is None:
        b_max = np.max(f.x)
    max_iter = int(max_range / step)
    sign_a = np.sign(f(a))
    for i in range(1, max_iter + 1):
        b_now = min(a + i * step, b_max)
        if np.sign(f(b_now)) == sign_a:
            continue
        root = brentq(f, a, b_now)
        return root
    return np.nan


def find_nearest_smaller_max(func, x, lower_bound=0, num_points=10000):
    from scipy.signal import argrelextrema
    x_vals = np.linspace(lower_bound, x, num_points, endpoint=False)
    y_vals = func(x_vals)
    # find all maxima
    max_indices = argrelextrema(y_vals, np.greater, order=1)[0]
    if len(max_indices) == 0:
        return None
    max_points = x_vals[max_indices]
    nearest_max = max_points[np.argmax(max_points)]
    return nearest_max


def curve_width(b, x, y):
    from scipy.optimize import bisect
    from scipy.interpolate import interp1d
    """
    For a curve with local maximum, given y=b, find the intercept width
    """
    max_idx = np.argmax(y)
    y_max = y[max_idx]

    if b >= y_max:
        return 0.0

    # 处理左侧（单调递增）
    x_left = x[:max_idx + 1]
    y_left = y[:max_idx + 1]
    f_left = interp1d(x_left, y_left - b, kind='linear')
    try:
        x1 = bisect(f_left, x_left[0], x_left[-1], xtol=1e-7)
    except ValueError:
        x1 = x_left[-1]  # 处理边界情况

    # 处理右侧（单调递减）
    x_right = x[max_idx:]
    y_right = y[max_idx:]
    f_right = interp1d(x_right, y_right - b, kind='linear')
    try:
        x2 = bisect(f_right, x_right[0], x_right[-1], xtol=1e-7)
    except ValueError:
        x2 = x_right[0]  # 处理边界情况

    return x2 - x1


def find_b_from_w(w_target, x, y):
    from scipy.optimize import root_scalar
    """
    For a curve with local maximum, given curve width, find y=b
    """
    y0 = np.max(y)
    max_idx = np.argmax(y)
    y_left = y[:max_idx + 1]
    y_right = y[max_idx:]
    b_low = max([np.min(y_left), np.min(y_right)])

    # 计算最大可能宽度w_max
    w_max = curve_width(b_low, x, y)

    # 处理特殊情况
    if w_target == 0:
        return y0
    if w_target < 0 or w_target > w_max:
        raise ValueError(f"目标宽度需在0到{w_max:.3g}之间，当前输入为{w_target}.")

    # 定义目标函数
    def objective(b):
        return curve_width(b, x, y) - w_target

    # 使用Brent方法求解（适用于单调函数）
    sol = root_scalar(
        objective,
        bracket=[b_low, y0],
        method='brentq',
        xtol=1e-8
    )

    if sol.converged:
        return sol.root
    else:
        raise RuntimeError("求解失败，请检查输入或采样点范围.")


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    # print(proj(np.array([1, 2, 3]), np.array([[5, 6, 2], [5, 6, 3]])))
    # np.random.seed(42)
    # x = np.sort(np.random.uniform(0, 10, 100))
    # y = np.sin(x) + np.random.normal(0, 0.2, len(x))
    # slope_estimate(x, y)

    # t = np.arange(0, 10, 2)
    # d1 = 2 * t ** 3 - 0.5 * t ** 4
    # d2 = 3 * t ** 3 - 0.3 * t ** 4
    # fs = time_integrate(t, np.vstack((d1, d2)))
    # plt.plot(t, fs.T)
    # plt.show()

    print(find_clusters([1, 2, 5, 6, 8, 12], 3))

    # print(root_nearest_smaller(lambda x: np.sin(np.pi * x), np.pi / 2))
    # def func(x):
    #     return np.sin(x)

    # print(find_nearest_smaller_max(func, x=3))
