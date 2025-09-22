from my_work.network_models.Motility_Model_noICC_diagnostics import MotilityModel
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from brian2.units import ms, mV, nA
from utils.result_path import RES_PATH

plt.rc('font', size=20, family='Arial')
font_label = {'size': 23, 'family': 'Arial'}
plt.rc('lines', lw=2)

case_name = 'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
case_path = f'{RES_PATH}/{case_name}'
net = MotilityModel(
    go_on=True, neuron_pop=92,
    force_factor=6e-5,
    N_callback_net=25, dt_couple=1e-3
)
net.restore(filename=f'{case_path}/store/SN_0to1250000.store')
time = net['mSN'].t / ms
i_ECMN = 10 + 92 * 2
fig, ax = plt.subplots(2, 1, sharex=True, layout='constrained', figsize=(16, 8))
ax[0].plot(time, net['mSN'].v[i_ECMN] / mV)
ax[0].plot([12085, 12087], [-20, -20], 'k')
ax[0].plot([12085, 12085], [-20, 10], 'k')
ax[0].axhline(y=-70, color='gray', ls='--')
ax[1].plot(time, net['mSN'].I_Nav13[i_ECMN] / nA, label='$\mathrm{Na_v1.3}$')
ax[1].plot(time, net['mSN'].I_Kdr[i_ECMN] / nA, label='Kdr')
ax[1].plot(time, net['mSN'].I_Kv72[i_ECMN] / nA, label='$\mathrm{K_v7.2}$')
ax[1].plot(time, net['mSN'].I_leak[i_ECMN] / nA, label='leak')
ax[1].plot(time, net['mSN'].I_EPSP[i_ECMN] / nA, label='EPSP')
ax[1].legend(loc='lower right', bbox_to_anchor=(0.8, 0.05), prop=font_label)
ax[1].plot([12085, 12085], [-600, -100], 'k')
ax[1].axhline(y=0, color='gray', ls='--')
ax[1].set_xlim([12080, 12120])
inset_I = ax[1].inset_axes(
    [12092, -1000, 9, 850],
    transform=ax[1].transData,
    xlim=(12086, 12095),
    ylim=(-17, 13)
)
inset_I.plot(time, net['mSN'].I_Nav13[i_ECMN] / nA)
inset_I.plot(time, net['mSN'].I_Kdr[i_ECMN] / nA)
inset_I.plot(time, net['mSN'].I_Kv72[i_ECMN] / nA)
inset_I.plot(time, net['mSN'].I_leak[i_ECMN] / nA)
inset_I.plot(time, net['mSN'].I_EPSP[i_ECMN] / nA)
inset_I.plot([12088, 12088], [-14, -4], 'k')
inset_I.plot([12088, 12090], [-14, -14], 'k')
inset_I.axhline(y=0, color='gray', ls='--')
mark_inset(ax[1], inset_I, loc1=1, loc2=3, ec='0.2', zorder=5)

for cur_ax in [ax[0], ax[1], inset_I]:
    cur_ax.set_xticks([])
    cur_ax.set_xticklabels([])
    cur_ax.set_yticks([])
    cur_ax.set_yticklabels([])
    if cur_ax is not inset_I:
        for spine in cur_ax.spines.values():
            spine.set_visible(False)
fig.savefig(f'{case_path}/SN-details.png', dpi=300)
plt.show()
