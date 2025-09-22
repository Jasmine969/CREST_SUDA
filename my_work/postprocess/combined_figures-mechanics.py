import matplotlib.pyplot as plt
import matplotlib as mpl
from visual_sph import flow_rate_map
from visualize_interface import strain_map, case_path
import numpy as np

mpl.rcParams['svg.fonttype'] = 'none'
plt.rc('font', size=17, family='Arial')
font_label = {'size': 20, 'family': 'Arial'}
# fig, ax = plt.subplots(4, 1, figsize=(5, 10), height_ratios=[1.3, 1.3, 1., 1.3])
fig, ax = plt.subplot_mosaic(
    [['strain', 'flow-rate']],
    figsize=np.array([9, 5]) * 1.5,
    layout='constrained'
)
xtick_s, xticklabel_s, ytick_s, yticklabel_s = strain_map(ax['strain'], hlines=[0.6,9.4, 11.5, 20.0])
ax['strain'].set_xticks(xtick_s)
ax['strain'].set_xticklabels(xticklabel_s)
ax['strain'].set_yticks(ytick_s)
ax['strain'].set_yticklabels(yticklabel_s)
ax['strain'].set_xlabel('x (mm)', fontdict=font_label)
ax['strain'].set_ylabel('Time (s)', fontdict=font_label)

xtick_r, xticklabel_r, ytick_r, yticklabel_r = flow_rate_map(ax['flow-rate'], hlines=[0.6,9.4, 11.5, 20.0])
ax['flow-rate'].set_xticks(xtick_r)
ax['flow-rate'].set_xticklabels(xticklabel_r)
ax['flow-rate'].set_yticks(ytick_r)
ax['flow-rate'].set_yticklabels(yticklabel_r)
ax['flow-rate'].set_xlabel('x (mm)', fontdict=font_label)

# import numpy as np
# im = ax['tmp'].imshow(np.linspace(0, 25, 50000).reshape(250, 200), aspect='auto', cmap='plasma')
# cb = plt.colorbar(im, orientation='horizontal')
# cb.set_label('Fluid velocity (mm/s)', fontdict=font_label)
# cb.set_ticks(ticks=np.arange(0, 26, 5), labels=np.arange(0, 26, 5))

fig.savefig(f'{case_path}/strainMap+flowRate.png', dpi=300, transparent=True)
fig.savefig(f'{case_path}/strainMap+flowRate.svg')
plt.show()
