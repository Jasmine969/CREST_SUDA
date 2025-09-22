import matplotlib.pyplot as plt
from visual_sph import draw_atom_force_arrow

plt.rc('font', size=17, family='Arial')
font_label = {'size': 20, 'family': 'Arial'}
fig, ax = plt.subplot_mosaic(
    [['frame80', 'frame88', 'frame217'], ],
    # ['force-time', 'force-time','force-time'],
    # ['strain-time', 'strain-time','strain-time']],
    # height_ratios = [2, 1, 0.5],
    layout='constrained'
)
draw_atom_force_arrow(80, ax['frame80'], if_annotate=False)
draw_atom_force_arrow(88, ax['frame88'], if_annotate=False)
draw_atom_force_arrow(217, ax['frame217'], if_annotate=False)
for name in ['frame80', 'frame88', 'frame217']:
    ax[name].set_xticks([])
    ax[name].set_xticklabels([])
    ax[name].set_yticks([])
    ax[name].set_yticklabels([])
    for spine in ax[name].spines.values():
        spine.set_visible(False)
plt.show()
