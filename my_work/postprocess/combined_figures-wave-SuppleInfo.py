import matplotlib.pyplot as plt
from visualize_interface import case_path, strain_map
from scipy.signal import convolve2d
import pickle
import numpy as np

font_ticks = {'size': 17, 'family': 'Arial'}
font_label = {'size': 20, 'family': 'Arial'}
plt.rc('font', **font_ticks)
fig, ax = plt.subplot_mosaic(
    [['strain', 'Dt'],
     ['DDt', 'lasso'],
     ['quiver', 'wave-vel']],
    layout='constrained',
    figsize=(12, 11),
    sharex=True
)

strain = np.load(f'{case_path}/interface/strain_tension_1250000.npz')['strain']
xticks, xticklabels, yticks, yticklabels = strain_map(ax['strain'])

dL = 2e-4 * 1e3  # mm
dt = 1e-3
Ncallback_lmp = 50
dt_lmp = 2e-5
sigma_t = 100 / 6
n_time_half = int(sigma_t * 3)
interval_arrow = 200
markersize = 2
scale = int(4000 / interval_arrow)
t = np.arange(-n_time_half, n_time_half + 1)
G_t = -t / (np.sqrt(2 * np.pi) * sigma_t ** 3 * dt) * np.exp(-t ** 2 / (2 * sigma_t ** 2))
G_t = G_t[:, np.newaxis]
strain_dt = convolve2d(strain, G_t, mode='same', boundary='symm')
strain_dtdt = convolve2d(strain_dt, G_t, mode='same', boundary='symm')
contraction_line = np.where((np.abs(strain_dt) < 0.07) & (strain_dtdt > 0) & (strain < -0.07))
contraction_line = np.array(contraction_line).T
contraction_line[:, 0] = contraction_line[:, 0] + 1  # time index of 0 on the strain_map is actually 1 ms
ax['strain'].plot(contraction_line[:, 1], contraction_line[:, 0], 'g.', markersize=markersize)
im_Dt = ax['Dt'].imshow(strain_dt, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
cb_Dt = plt.colorbar(im_Dt)
cb_Dt.set_ticks(np.arange(-1, 1.1, 0.5))
cb_Dt.set_label(r'$\partial\bar{\varepsilon}/\partial t~(\mathrm{1/s})$ ', fontdict=font_label)
ax['Dt'].plot(contraction_line[:, 1], contraction_line[:, 0], 'g.', markersize=markersize)
im_DDt = ax['DDt'].imshow(strain_dtdt, cmap='bwr', aspect='auto', vmin=-10, vmax=10)
cb_DDt = plt.colorbar(im_DDt)
cb_DDt.set_ticks(np.arange(-10, 11, 5))
cb_DDt.set_label(r'$\partial^2\bar{\varepsilon}/\partial t^2~(\mathrm{1/s^2})$ ', fontdict=font_label)
ax['DDt'].plot(contraction_line[:, 1], contraction_line[:, 0], 'g.', markersize=markersize)

ax['lasso'].plot(contraction_line[:, 1], contraction_line[:, 0], 'g.', markersize=markersize)
with open(f'{case_path}/wave_labels.pkl', 'rb') as pf:
    groups = pickle.load(pf)
cmap = plt.get_cmap('cividis')
colors = cmap(np.linspace(0, 0.9, len(groups)))
for i, each in enumerate(groups):
    ax['lasso'].plot(each[:, 1], each[:, 0], '.',color=colors[i], markersize=markersize*1.2)
ax['lasso'].invert_yaxis()
with open(f'{case_path}/slope_scale_tx_smooth_list.pkl', 'rb') as pf2:
    slope_scale_tx_smooth_list = pickle.load(pf2)

for i, slope_scale_tx_smooth in enumerate(slope_scale_tx_smooth_list):
    uv = np.c_[slope_scale_tx_smooth[:, 0], np.ones(slope_scale_tx_smooth.shape[0])]
    uv /= (uv ** 2).sum(axis=1, keepdims=True) ** 0.5
    ax['quiver'].quiver(
        slope_scale_tx_smooth[::interval_arrow, 3], slope_scale_tx_smooth[::interval_arrow, 2],
        uv[::interval_arrow, 0], uv[::interval_arrow, 1],
        scale=scale, angles='xy', color=colors[i],
        width=0.006, headwidth=5, headlength=5, headaxislength=4.5,
        linewidth=2
    )
    ax['wave-vel'].plot(slope_scale_tx_smooth[:, 3], slope_scale_tx_smooth[:, 1], lw=2, color=colors[i])

for name in ax.keys():
    if name == 'wave-vel':
        continue
    ax[name].set_yticks(yticks)
    ax[name].set_yticklabels(yticklabels)
    ax[name].set_ylabel('Time (s)', fontdict=font_label)
    ax[name].set_ylim(yticks[-1], yticks[0])
for name in ax.keys():
    if name in ['quiver', 'wave-vel']:
        ax[name].set_xticks(xticks)
        ax[name].set_xticklabels(xticklabels)
        ax[name].set_xlabel('x (mm)', fontdict=font_label)
ax['wave-vel'].set_xlim([0, 199])
ax['wave-vel'].set_ylabel('Wave velocity (mm/s)', fontdict=font_label)

vel_all = np.vstack(slope_scale_tx_smooth_list)[:,1]
print(f'Vel={vel_all.mean()}±{vel_all.std()} mm/s, n={vel_all.size}')
fig.savefig(f'{case_path}/wave-analysis.png', dpi=300)
plt.show()
