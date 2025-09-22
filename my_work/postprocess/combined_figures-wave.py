import matplotlib.pyplot as plt
from visualize_interface import wave_vel_paper, interpolate_ring_strain
from visual_sph import interpolate_FSI_force, case_path
import numpy as np
import pandas as pd

font_ticks = {'size': 17, 'family': 'Arial'}
font_label = {'size': 20, 'family': 'Arial'}
plt.rc('font', **font_ticks)
fig, ax = plt.subplot_mosaic(
    [['quiver', 'wave-vel'],
     # ['same-x-force', 'front-force'],
     # ['same-x-strain', 'front-strain']
     ],
    layout='constrained',
    figsize=(14, 6),
    # height_ratios=[1.5, 1, 1]
)
id_front = [0, 3]
xtick_q, xticklabel_q, ytick_q, yticklabel_q, fronts_same_wave, peaks_same_x = wave_vel_paper(
    ax_map=ax['quiver'], ax_vel=ax['wave-vel'],
    # need_front_same_wave=id_front, need_peak_same_x=[50]
)
ax['quiver'].set_xlabel('x (mm)', fontdict=font_label)
ax['quiver'].set_ylabel('Time (s)', fontdict=font_label)
ax['wave-vel'].set_xlim([0, 199])
ax['wave-vel'].set_xlabel('x (mm)', fontdict=font_label)
ax['wave-vel'].set_ylabel('Wave velocity (mm/s)', fontdict=font_label)
ax['quiver'].set_xticks(xtick_q)
ax['quiver'].set_xticklabels(xticklabel_q)
ax['quiver'].set_yticks(ytick_q)
ax['quiver'].set_yticklabels(yticklabel_q)
ax['wave-vel'].set_xticks(xtick_q)
ax['wave-vel'].set_xticklabels(xticklabel_q)
# cmap = plt.get_cmap('cividis')
# colors = cmap(np.linspace(0, 0.9, 7))
# # plot_force_strain_ring(ax['same-x-force'], names_force=['FSI'],
# #                        which_strain='interface', positive_force='outward')
# ringID = 50
# df_force = pd.read_excel(f'{case_path}/force-ring{ringID}.xlsx', usecols=['FSI'])
# df_force['strain'] = np.r_[
#     np.load(f'{case_path}/all_rings_strain.npy')[ringID, 0],
#     np.load(f'{case_path}/interface/strain_tension_1250000.npz'
#             )['strain'].T[ringID, 99::100]
# ]
# df_force['time'] = np.arange(df_force.shape[0]) * 0.1
# for i in range(len(peaks_same_x)):
#     if i == 0:
#         mask = df_force['time'] * 1000 <= peaks_same_x[i]
#     else:
#         mask = (df_force['time'] * 1000 >= peaks_same_x[i - 1]) & (
#                 df_force['time'] * 1000 <= peaks_same_x[i])
#     i_next = np.where(mask)[0].max() + 1
#     if i_next < df_force.shape[0]:
#         mask[i_next] = True
#     cur_data = df_force.loc[mask]
#     ax['same-x-force'].plot(cur_data['time'], -cur_data['FSI'], c=colors[i])
#     ax['same-x-strain'].plot(cur_data['time'], cur_data['strain'], c=colors[i])
# ax['same-x-force'].set_xticks([0, 5, 10, 15, 20, 25])
# ax['same-x-force'].set_xticklabels([''] * 6)
# ax['same-x-strain'].set_xticks([0, 5, 10, 15, 20, 25])
# ax['same-x-strain'].set_yticks([0.06, 0.09, 0.12, 0.15])
# ax['same-x-force'].set_xlim([0, 25])
# ax['same-x-strain'].set_xlim([0, 25])
# ax['same-x-strain'].set_ylim([0.06, 0.15])
# ax['same-x-strain'].set_xlabel('Time (s)', fontdict=font_label)
# ax['same-x-force'].set_ylabel('Outward FSI\nforce (μN)', fontdict=font_label)
# ax['same-x-strain'].set_ylabel('Strain', fontdict=font_label)
# for front, i in zip(fronts_same_wave, id_front):
#     FSI_force = interpolate_FSI_force(front)
#     ax['front-force'].plot(front[:, 1], -FSI_force, c=colors[i])
#     strain = interpolate_ring_strain(front)
#     ax['front-strain'].plot(front[:, 1], strain, c=colors[i])
# # ax['front-force'].set_ylabel('Outward FSI\nforce (μN)', fontdict=font_label)
# ax['front-strain'].set_xlabel('x (mm)', fontdict=font_label)
# ax['front-strain'].set_ylabel('Strain', fontdict=font_label)
# ax['front-force'].set_ylabel('Outward FSI\nforce (μN)', fontdict=font_label)
# ax['front-force'].set_xticks(xtick_q)
# ax['front-force'].set_xticklabels([''] * len(xticklabel_q))
# ax['front-strain'].set_xticks(xtick_q)
# ax['front-strain'].set_xticklabels(xticklabel_q)
# ax['front-strain'].set_yticks([0.0, 0.05, 0.1, 0.15])
# ax['front-force'].set_xlim([0, 199])
# ax['front-strain'].set_xlim([0, 199])
fig.savefig(f'{case_path}/wave-analysis.png', dpi=300)
plt.show()
