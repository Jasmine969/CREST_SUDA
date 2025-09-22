import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from visualize_interface import (
    plot_strain_troughs, plot_tension_peaks, plot_strain_multiframe, plot_tension_multiframe)
from visualize_network import plot_activation_window
from visual_sph import plot_longitudinal_bond_force_multiframe, case_path

font_ticks = {'size': 17, 'family': 'Arial'}
font_label = {'size': 20, 'family': 'Arial'}
mpl.rcParams['svg.fonttype'] = 'none'
plt.rc('font', **font_ticks)
fig1, ax1 = plt.subplot_mosaic(
    [['peak-strain'],
     ['peak-tension'],
     ['peak-window']],
    layout='constrained', figsize=(6, 8)
)
fig2, ax2 = plt.subplot_mosaic(
    [['region-strain'],
     ['region-tension'],
     ['region-ve']],
    layout='constrained', figsize=(6, 8)
)
plot_strain_troughs(waveID_global=5, ax=ax1['peak-strain'], use_reverse_waveID=True)
ax1['peak-strain'].set_ylabel('Strain', fontdict=font_label)
plot_tension_peaks(waveID_global=5, ax=ax1['peak-tension'], use_reverse_waveID=True)
ax1['peak-tension'].set_ylabel('Active force (μN)', fontdict=font_label)
plot_activation_window(waveID_global=5, ax=ax1['peak-window'], use_reverse_waveID=True)
ax1['peak-window'].set_ylabel('Activation time (s)', fontdict=font_label)
ax1['peak-window'].set_xlabel('Ring ID', fontdict=font_label)
xtick_p = np.arange(10, 161, 30)
for name, ax_ in ax1.items():
    if 'region' in name:
        continue
    ax_.set_xticks(xtick_p)
    if 'window' in name:
        ax_.set_xticklabels(xtick_p)
    else:
        ax_.set_xticklabels([''] * len(xtick_p))
ringIDmin, ringIDmax = 25, 37
plot_strain_multiframe(frames=list(range(17400 - 1, 18000, 100)),
                       ringIDmin=ringIDmin, ringIDmax=ringIDmax, ax=ax2['region-strain'])
plot_tension_multiframe(frames=list(range(17400 - 1, 18000, 100)),
                        ringIDmin=ringIDmin, ringIDmax=ringIDmax, ax=ax2['region-tension'])
plot_longitudinal_bond_force_multiframe(list(range(174, 181, 1)),
                                        ringIDmin=ringIDmin, ringIDmax=ringIDmax, ax=ax2['region-ve'])
ax2['region-strain'].axhline(0.068, ls='--', color='gray')
ax2['region-strain'].set_ylabel('Strain', fontdict=font_label)
ax2['region-tension'].set_ylabel('Active force (μN)', fontdict=font_label)
ax2['region-ve'].set_ylabel('Longitudinal\nbond force (μN)', fontdict=font_label)
ax2['region-ve'].set_xlabel('Ring ID', fontdict=font_label)
xticklabel_r = np.arange(ringIDmin, ringIDmax, 2)
xtick_r = xticklabel_r - ringIDmin
for name, ax_ in ax2.items():
    if 'peak' in name:
        continue
    ax_.set_xlim([-0.5, ringIDmax-ringIDmin - 0.5])
    ax_.set_xticks(xtick_r)
    if 've' in name:
        ax_.set_xticklabels(xticklabel_r)
    else:
        ax_.set_xticklabels([''] * len(xticklabel_r))
fig1.savefig(f'{case_path}/saltatory-analysis1.png', dpi=300)
fig1.savefig(f'{case_path}/saltatory-analysis1.svg')
fig2.savefig(f'{case_path}/saltatory-analysis2.png', dpi=300)
fig2.savefig(f'{case_path}/saltatory-analysis2.svg')
plt.show()
