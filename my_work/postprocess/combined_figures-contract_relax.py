from literature_plot import *
from visualize_interface import contraction_relaxation_vel, strain_single_ring
from visual_sph import plot_force_strain_ring, case_path
import matplotlib.pyplot as plt
import matplotlib as mpl

font_ticks = {'size': 17, 'family': 'Arial'}
font_label = {'size': 20, 'family': 'Arial'}
mpl.rcParams['svg.fonttype'] = 'none'
plt.rc('font', **font_ticks)
plt.rc('lines', lw=2)

fig, ax = plt.subplot_mosaic(
    [['JP2004', 'mine', 'comparison'],
     ['forces', 'forces', 'forces'],
     ['strain', 'strain', 'strain']],
    layout='constrained',
    figsize=(12, 10),
    # width_ratios=[1.6, 1.6, 0.5, 0.5]
)
plot_JP2004(ax['JP2004'])
ax['JP2004'].set_xticks(list(range(0,11,2)))
ax['JP2004'].set_xticklabels(list(range(0,11,2)), fontdict=font_label)
ax['JP2004'].set_xlabel('Time (s)', fontdict=font_label)
ax['JP2004'].set_ylabel('Diameter (mm)', fontdict=font_label)
ax['JP2004'].set_xlim([0, 10])
ax['JP2004'].set_ylim([2, 5])
strain_single_ring(50, diameter=True, ax=ax['mine'])
ax['mine'].axhline(4,color='gray', ls='--')
ax['mine'].set_xlabel('Time (s)', fontdict=font_label)
ax['mine'].set_ylabel('Diameter (mm)', fontdict=font_label)
ax['mine'].set_xlim([5080, 6640])
ax['mine'].set_xticks([5200, 5600, 6000, 6400])
ax['mine'].set_xticklabels([5.2, 5.6, 6.0, 6.4], fontdict=font_label)
contraction_relaxation_vel(ax['comparison'])
ax['comparison'].set_xlim([0, 5])
ax['comparison'].set_ylim([0, 5])
ticks_r = list(range(6))
ax['comparison'].set_xticks(ticks_r)
ax['comparison'].set_xticklabels(ticks_r)
ax['comparison'].set_yticks(ticks_r)
ax['comparison'].set_yticklabels(ticks_r)
ax['comparison'].set_ylabel('Contraction rate (mm/s)', fontdict=font_label)
ax['comparison'].set_xlabel('Relaxation rate (mm/s)', fontdict=font_label)
plot_force_strain_ring(ringID=50, ax_force=ax['forces'], ax_strain=ax['strain'])
# ax['strain'].axhline(0.068, ls='--', color='gray')
ax['forces'].set_ylabel('Forces (μN)', fontdict=font_label)
xtick_f = list(range(0, 26, 5))
ax['forces'].set_xticks(xtick_f)
ax['forces'].set_xticklabels([''] * len(xtick_f))
ax['strain'].set_ylim([-0.33, 0.15])
ax['strain'].set_xlabel('Time (s)', fontdict=font_label)
ax['strain'].set_ylabel('Strain', fontdict=font_label)
fig.savefig(f'{case_path}/contract_relax.png', dpi=300)
fig.savefig(f'{case_path}/contract_relax.svg')
plt.show()
