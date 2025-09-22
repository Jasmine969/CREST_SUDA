import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils.id2x import x2SIP_ID
from visualize_interface import tension_map, case_path
from visualize_network import calcium_map

font_ticks = {'size': 17, 'family': 'Arial'}
font_label = {'size': 20, 'family': 'Arial'}
mpl.rcParams['svg.fonttype'] = 'none'
plt.rc('font', **font_ticks)
fig, ax = plt.subplot_mosaic(
    [['tension'],
     ['Ca']],
    layout='constrained', figsize=np.array([5, 7.5]) * 1.1
)
mark_x = 13.4
mark_SMC = x2SIP_ID(mark_x)
xtick_t, xticklabel_t, ytick_t, yticklabel_t = tension_map(ax['tension'], vline=mark_SMC)
ax['tension'].set_xticks(xtick_t)
ax['tension'].set_xticklabels([''] * len(xticklabel_t))
ax['tension'].set_yticks(ytick_t)
ax['tension'].set_yticklabels(yticklabel_t)
ax['tension'].set_ylabel('Time (s)', fontdict=font_label)
xtick_c, xticklabel_c, ytick_c, yticklabel_c = calcium_map(ax['Ca'], vline=mark_SMC)
ax['Ca'].set_xticks(xtick_c)
ax['Ca'].set_xticklabels(xticklabel_c)
ax['Ca'].set_yticks(ytick_c)
ax['Ca'].set_yticklabels(yticklabel_c)
ax['Ca'].set_xlabel('x (mm)', fontdict=font_label)
ax['Ca'].set_ylabel('Time (s)', fontdict=font_label)
fig.savefig(f'{case_path}/tension+Ca.png', dpi=300)
fig.savefig(f'{case_path}/tension+Ca.svg')
plt.show()
