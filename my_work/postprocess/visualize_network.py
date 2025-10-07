"""
SIPID is SMC ID here.
"""
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm, colormaps
from functools import partial
from tqdm import tqdm
import os
import numpy as np
from my_work.network_models.Motility_Model_noICC_diagnostics import MotilityModel
from brian2.units import ms, mV, nA, uM
import pickle
from utils.result_path import RES_PATH

mpl.rcParams['svg.fonttype'] = 'none'
font_ticks = {'size': 17, 'family': 'Arial'}
font_label = {'size': 20, 'family': 'Arial'}
Ncallback_lmp = 50
case_name = 'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
case_path = f'{RES_PATH}/{case_name}'
go_on_from_step = 0
read_step = 1250000
with open(f'{case_path}/net_params.pkl', 'rb') as f:
    net_params = pickle.load(f)
net = MotilityModel(False, **net_params)
# with open(f'{case_path}/state/net_{read_step}.pkl', 'rb') as pf_state:
#     net.set_states(pickle.load(pf_state))
net.restore(filename=f'{case_path}/store/net_{go_on_from_step}to{read_step}.store')
# net.restore(filename=f'{case_path}/store/8520.store')
# net.restore(filename=f'{RES_PATH}/default/store/test.store')
dt_lmp = 2e-5
my_ratio = 200
neuron_pop, muscle_pop = int(len(net['SN'].v) / 4), len(net['SMC'].v)

t = net['mSMC'].t / ms
step = t / 1000 / dt_lmp
time_Tavg = net['mTavg'].t / ms
TSMC = net['mTavg'].T_avg * 1e6
vSMC = net['mSMC'].v / mV
# vICC = net['mICC'].v / mV
vSN = net['mSN'].v / mV
vAIN = vSN[:neuron_pop]
vDIN = vSN[neuron_pop:neuron_pop * 2]
vECMN = vSN[neuron_pop * 2:neuron_pop * 3]
vICMN = vSN[neuron_pop * 3:neuron_pop * 4]
vIPAN = net['mIPAN'].v / mV
DSTND = net['mStrain'].DSTND
t_spikeIPAN = [each / ms for each in net['sIPAN'].values('t').values()]
t_spikeSN = [each / ms for each in net['sSN'].values('t').values()]
t_spikeAIN = t_spikeSN[:neuron_pop]
t_spikeDIN = t_spikeSN[neuron_pop:neuron_pop * 2]
t_spikeECMN = t_spikeSN[neuron_pop * 2:neuron_pop * 3]
t_spikeICMN = t_spikeSN[neuron_pop * 3:]


# np.save(f'{case_path}/distension', DSTND)
#
# import pickle
#
# with open(f'{case_path}/tspikes.pkl', 'wb') as f:
#     pickle.dump([t_spikeSN[key] / ms for key in range(92 + 72, 92 + 76)], f)


def visualize_dynamics(draw_ICC):
    """To make the network part of SV2-couple.mp4"""
    figure_path = f'{case_path}/net-snapshot'
    os.makedirs(figure_path, exist_ok=True)
    x_offset, x_scale = 1, 5
    delta_y = 0.3
    IPAN_x, IPAN_y = np.arange(neuron_pop) * x_scale, np.zeros(neuron_pop)
    AIN_x, AIN_y = IPAN_x - x_offset, IPAN_y + delta_y
    DIN_x, DIN_y = IPAN_x + x_offset, AIN_y
    ECMN_x, ECMN_y = AIN_x, IPAN_y + 2 * delta_y
    ICMN_x, ICMN_y = DIN_x, ECMN_y
    SMC_x, SMC_y = (0.5 * np.arange(muscle_pop) - 0.25) * x_scale, np.zeros(muscle_pop) + 3 * delta_y
    GapSMC_i, GapSMC_j = list(net['GJ_SMC_SMC'].i), list(net['GJ_SMC_SMC'].j)
    if draw_ICC:
        SMC_y += delta_y
        GapISMC_i, GapISMC_j = list(net['GJ_ICC_SMC'].i), list(net['GJ_ICC_SMC'].j)
        GapICC_i, GapICC_j = list(net['GJ_ICC_ICC'].i), list(net['GJ_ICC_ICC'].j)
        ICMN_ICC_i, ICMN_ICC_j = list(net['ICMN_ICC'].i), list(net['ICMN_ICC'].j)
        ICC_x, ICC_y = SMC_x + (np.arange(1, 1 + muscle_pop) % 2 - 0.5) * 0.7, SMC_y - delta_y
    ECMN_SMC_i, ECMN_SMC_j = list(net['ECMN_SMC'].i), list(net['ECMN_SMC'].j)
    ICMN_SMC_i, ICMN_SMC_j = list(net['ICMN_SMC'].i), list(net['ICMN_SMC'].j)
    AIN_ECMN_i, AIN_ECMN_j = list(net['AIN_ECMN'].i), list(net['AIN_ECMN'].j)
    DIN_ICMN_i, DIN_ICMN_j = list(net['DIN_ICMN'].i), list(net['DIN_ICMN'].j)
    IPAN_AIN_i, IPAN_AIN_j = list(net['IPAN_AIN'].i), list(net['IPAN_AIN'].j)
    IPAN_DIN_i, IPAN_DIN_j = list(net['IPAN_DIN'].i), list(net['IPAN_DIN'].j)
    w_AIN_ECMN = net['AIN_ECMN'].w
    w_DIN_ICMN = net['DIN_ICMN'].w
    msize = 20
    # vmax is set to threshold
    VNmin, VNmax = -68, 30
    norm = cm.colors.Normalize(vmin=-65, vmax=10)
    normICC = cm.colors.Normalize(vmin=-65, vmax=-21)
    normSMC = cm.colors.Normalize(vmin=-75, vmax=-30)
    normT = cm.colors.Normalize(vmin=0, vmax=10)
    normDSTND = cm.colors.Normalize(vmin=-0.4, vmax=0.2)
    scatter = partial(plt.scatter, s=msize, cmap='coolwarm')
    plt.figure(figsize=(26.7, 8))
    ec = 'w'
    font_legend = {'size': 15, 'family': 'Arial'}
    font_text = {'size': 12, 'family': 'Arial'}
    linewidth = 1.5
    linecolor = 'C2'
    transparency = 0.6
    x_offset = 0.5
    y_offset = delta_y * 5e-2

    IPAN_y -= delta_y * 0.15
    SMC_y += delta_y * 0.15
    # draw the first frame
    tind = 0
    sc_ECMN = scatter(ECMN_x, ECMN_y, c=vECMN[:, tind], marker='s', edgecolor=ec, norm=norm)
    sc_ICMN = scatter(ICMN_x, ICMN_y, c=vICMN[:, tind], marker='s', edgecolor=ec, norm=norm)
    sc_AIN = scatter(AIN_x, AIN_y, c=vAIN[:, tind], marker='s', edgecolor=ec, norm=norm)
    sc_DIN = scatter(DIN_x, DIN_y, c=vDIN[:, tind], marker='s', edgecolor=ec, norm=norm)
    sc_IPAN = scatter(IPAN_x, IPAN_y, c=vIPAN[:, tind], marker='s', edgecolor=ec, norm=norm)
    sc_SMC = scatter(SMC_x, SMC_y, c=vSMC[:, tind], marker='o', edgecolor=ec, norm=normSMC)
    sc_T = plt.scatter(SMC_x, SMC_y + delta_y * 0.15, msize, c=TSMC[:, tind // 2],
                       marker='o', edgecolor=ec, cmap='coolwarm', norm=normT)
    sc_DSTND = plt.scatter(IPAN_x, IPAN_y - delta_y * 0.15, msize, c=DSTND[:, tind // 2],
                           marker='s', edgecolor=ec, cmap='coolwarm', norm=normDSTND)
    if draw_ICC:
        scatter(ICC_x, ICC_y, c=vICC[:, tind], marker='^', edgecolor=ec, norm=normICC)

    # colorbar ===========
    cb_pad = 0.03
    cb = plt.colorbar(sc_ECMN, orientation='horizontal',
                      fraction=0.036, aspect=9, pad=cb_pad)
    cb.set_label(r'Normalized $V$, $\bar{\varepsilon}$ or $F_\mathrm{a}$', fontdict=font_legend)
    cb.set_ticks([])
    plt.text(0.552, -cb_pad - 0.024, '1',
             horizontalalignment='center',  # 水平居中
             verticalalignment='center',  # 垂直居中
             transform=plt.gca().transAxes, fontdict=font_legend
             )
    plt.text(0.447, -cb_pad - 0.024, '0',
             horizontalalignment='center',  # 水平居中
             verticalalignment='center',  # 垂直居中
             transform=plt.gca().transAxes, fontdict=font_legend
             )

    for i, j, w in zip(AIN_ECMN_i, AIN_ECMN_j, w_AIN_ECMN):
        plt.plot([AIN_x[i], ECMN_x[j]], [AIN_y[i] + y_offset, ECMN_y[j] - y_offset], c=linecolor, lw=linewidth,
                 alpha=w * transparency)
    for i, j, w in zip(DIN_ICMN_i, DIN_ICMN_j, w_DIN_ICMN):
        plt.plot([DIN_x[i], ICMN_x[j]], [DIN_y[i] + y_offset, ICMN_y[j] - y_offset], c=linecolor, lw=linewidth,
                 alpha=w * transparency)
    for i, j in zip(IPAN_AIN_i, IPAN_AIN_j):
        plt.plot([IPAN_x[i], AIN_x[j]], [IPAN_y[i] * 1.03 + y_offset, AIN_y[j] - y_offset], c=linecolor,
                 lw=linewidth,
                 alpha=transparency)
    for i, j in zip(IPAN_DIN_i, IPAN_DIN_j):
        plt.plot([IPAN_x[i], DIN_x[j]], [IPAN_y[i] * 1.03 + y_offset, DIN_y[j] - y_offset], c=linecolor,
                 lw=linewidth,
                 alpha=transparency)
    for i, j in zip(ECMN_SMC_i, ECMN_SMC_j):
        plt.plot([ECMN_x[i], SMC_x[j]], [ECMN_y[i] + y_offset, SMC_y[j] - y_offset], c=linecolor, lw=linewidth,
                 alpha=transparency)
    for i, j in zip(ICMN_SMC_i, ICMN_SMC_j):
        plt.plot([ICMN_x[i], SMC_x[j]], [ICMN_y[i] + y_offset, SMC_y[j] - y_offset], c=linecolor, lw=linewidth,
                 alpha=transparency)
    for i, j in zip(GapSMC_i, GapSMC_j):
        xs = sorted([SMC_x[i], SMC_x[j]])
        plt.plot([xs[0] + x_offset, xs[1] - x_offset], [SMC_y[i], SMC_y[j]], '--',
                 c=linecolor,
                 alpha=transparency)
    if draw_ICC:
        for i, j in zip(GapISMC_i, GapISMC_j):
            plt.plot([ICC_x[i], SMC_x[j]], [ICC_y[i] + y_offset, SMC_y[j] - y_offset], '--', c=linecolor,
                     alpha=transparency)
        for i, j in zip(GapICC_i, GapICC_j):
            xs = sorted([ICC_x[i], ICC_x[j]])
            plt.plot([xs[0] + x_offset, xs[1] - x_offset], [ICC_y[i], ICC_y[j]], '--',
                     c=linecolor,
                     alpha=transparency)
    cur_time = t[tind] / 1000
    cur_step = round(cur_time / dt_lmp)
    plt.title(f't={cur_time:.2f} s',
              fontdict={'size': 20, 'family': 'Arial'})
    label_xcor = -20
    plt.text(label_xcor + 3, IPAN_y[0] - 0.1, r'$\bar{\varepsilon}$', fontdict=font_legend)
    plt.text(label_xcor, IPAN_y[0] - 0.02, 'IPAN', fontdict=font_legend)
    plt.text(label_xcor, AIN_y[0] - 0.07, 'AIN\nDIN', fontdict=font_legend)
    plt.text(label_xcor - 3.5, ECMN_y[0] - 0.03, 'ECMN\nICMN', fontdict=font_legend)
    if draw_ICC:
        plt.text(label_xcor, ICC_y[0] - 0.07, 'ICC-MY', fontdict=font_legend)
    plt.text(label_xcor - 1, SMC_y[0] - 0.03, 'SMC', fontdict=font_legend)
    plt.text(label_xcor + 2.5, SMC_y[0] + 0.03, '$F_\mathrm{a}$', fontdict=font_legend)
    # plt.xlim(-0.041, 0.63)
    plt.axis('off')
    for i in range(0, 10, 2):
        plt.text(IPAN_x[i], IPAN_y[i] - 0.1, str(i), fontdict=font_text, ha='center')
    for i in range(10, 92, 2):
        plt.text(IPAN_x[i], IPAN_y[i] - 0.1, str(i), fontdict=font_text, ha='center')
    for i in range(0, 10, 4):
        plt.text(SMC_x[i], SMC_y[i] + 0.06, str(i), fontdict=font_text, ha='center')
    for i in range(12, 184, 4):
        plt.text(SMC_x[i], SMC_y[i] + 0.06, str(i), fontdict=font_text, ha='center')
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches([15, 5])
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{figure_path}\\step_{cur_step:07d}.png', dpi=100)
    start = 20
    end = len(t)
    # end = int(len(t) * 0.2) + 1
    # start_timestep, end_timestep = 2000, 7000 + 1
    # start, end = (int(start_timestep * dt_lmp * 2000),
    #               int(end_timestep * dt_lmp * 2000))

    for tind in tqdm(range(start, end, 20)):
        sc_ECMN.set_array(vECMN[:, tind])
        sc_ICMN.set_array(vICMN[:, tind])
        sc_AIN.set_array(vAIN[:, tind])
        sc_DIN.set_array(vDIN[:, tind])
        sc_IPAN.set_array(vIPAN[:, tind])
        sc_SMC.set_array(vSMC[:, tind])
        sc_T.set_array(TSMC[:, tind // 2])
        sc_DSTND.set_array(DSTND[:, tind // 2])
        cur_time = t[tind] / 1000
        cur_step = round(cur_time / dt_lmp)
        plt.title(f't={cur_time:.2f} s',
                  fontdict={'size': 20, 'family': 'Arial'})
        plt.savefig(f'{figure_path}\\step_{cur_step:07d}.png', dpi=100)
        # plt.show()


def calc_freq(t_spike):
    # t_spike = t_spike[t_spike > 3600]
    fq = t_spike.size / ((t_spike[-1] - t_spike[0]) / 1000)
    return fq  # Hz


def draw_ICC():
    plt.rc('font', size=20, family='Arial')
    font_label = {'size': 23, 'family': 'Arial'}
    plt.rc('lines', lw=2)
    fig, ax = plt.subplot_mosaic(
        [['V'], ['I'], ['h'], ['cGMP']],
        layout='constrained', sharex=True)
    ids = [0, 10]
    ax['V'].plot(t, vICC[ids].T)
    ls = ['-', '--']
    for i, i_ICC in enumerate(ids):
        ax['I'].plot(t, net['mICC'].I_SOC[i_ICC] / nA, f'C0{ls[i]}', label='SOC')
        ax['I'].plot(t, net['mICC'].I_ANO1[i_ICC] / nA, f'C1{ls[i]}', label='ANO1')
        ax['I'].plot(t, net['mICC'].I_CaT[i_ICC] / nA, f'C2{ls[i]}', label='CaT')
        ax['I'].plot(t, net['mICC'].I_Kb[i_ICC] / nA, f'C3{ls[i]}', label='Kb')
        ax['I'].plot(t, net['mICC'].I_Nab[i_ICC] / nA, f'C4{ls[i]}', label='Nab')
        ax['I'].plot(t, net['mICC'].I_NSV[i_ICC] / nA, f'C5{ls[i]}', label='NSV')
        ax['I'].plot(t, net['mICC'].I_GJ_ICC_ICC[i_ICC] / nA, f'C6{ls[i]}', label='GJ_ICC_ICC')
        ax['I'].plot(t, net['mICC'].I_GJ_SMC_ICC[i_ICC] / nA, f'C7{ls[i]}', label='GJ_SMC_ICC')
    ax['I'].legend(loc='best', prop=font_label)
    ax['h'].plot(t, net['mICC'].h_ANO1[ids].T)
    ax['cGMP'].plot(t, net['mICC'].cGMP[ids].T)
    plt.show()


def raster_plot_tension(enlarge, colorbar=True):
    tension = TSMC * 1e6
    print(tension.max())
    plt.rc('font', **font_ticks)
    figsize = (12, 4.5) if enlarge else (12, 9)
    fig, ax1 = plt.subplots(figsize=figsize)
    ims = ax1.imshow(
        tension, aspect='auto',
        cmap='binary',
        vmin=0, vmax=10
    )
    if colorbar:
        cb = plt.colorbar(ims, pad=0.1, shrink=0.7, anchor=(0.0, 0.9))
        cb.set_label('Active force (μN)', fontdict=font_label)
    ax1.set_xlabel('Time (s)', fontdict=font_label)
    ax1.set_ylabel('x (mm)', fontdict=font_label)
    msize = 6 if enlarge else 2.5
    opacity = 0.8 if enlarge else 0.5
    delta_y = 0.2 if enlarge else 0.4
    ax2 = ax1.twinx()
    linelength = 0.17
    linewidth = 8
    # indices start from 0
    ax2.eventplot(t_spikeECMN, lineoffsets=np.arange(neuron_pop) - delta_y, colors='C0',
                  linelengths=linelength, linewidth=linewidth, alpha=opacity)
    ax2.eventplot(t_spikeICMN, lineoffsets=np.arange(neuron_pop), colors='C1',
                  linelengths=linelength, linewidth=linewidth, alpha=opacity)
    ax2.eventplot(t_spikeIPAN, lineoffsets=np.arange(neuron_pop) + delta_y, colors='C4',
                  linelengths=linelength, linewidth=linewidth, alpha=opacity)
    # ax2.plot(t_spikeECMN[:, 0], t_spikeECMN[:, 1] - delta_y, 'C0s', ms=msize, alpha=opacity)
    # ax2.plot(t_spikeICMN[:, 0], t_spikeICMN[:, 1], 'C1s', ms=msize, alpha=opacity)
    # ax2.plot(t_spikeIPAN[:, 0], t_spikeIPAN[:, 1] + delta_y, 'C4s', ms=msize, alpha=opacity)
    ax2.invert_yaxis()
    ax1.set_ylabel('SMC index', fontdict=font_label)
    ax2.set_ylabel('Neuron index', fontdict=font_label)
    if enlarge:
        neu_min, neu_max = 30 - 4, 30 + 2
        smc_min, smc_max = neu_min * 2 - 1, neu_max * 2
        t_min, t_max, dt = 2.6, 5., 0.4
        ax1.set_xlim(np.array([t_min, t_max]) * 1000)
        ax1.set_ylim([smc_max - 0.5, smc_min - 1.5])
        ax2.set_ylim([neu_max - 0.5, neu_min - 1.5])
        ax1.set_xticks(np.arange(t_min, t_max + dt * 0.5, dt) * 1000)
        ax1.set_xticklabels([f'{each:.1f}' for each in np.arange(t_min, t_max + dt, dt)], fontdict=font_ticks)
        ax1.set_yticks(np.arange(smc_min, smc_max + 1, 4) - 1)
        ax1.set_yticklabels(np.arange(smc_min, smc_max + 1, 4), fontdict=font_ticks)
        ax2.set_yticks(np.arange(neu_min, neu_max + 1, 2) - 1)
        ax2.set_yticklabels(np.arange(neu_min, neu_max + 1, 2), fontdict=font_ticks)
        filename = f'raster_STmap_enlarge_smc{smc_min}_{smc_max}.png'
    else:
        ax1.set_xlim([0, 10000])
        ax1.set_ylim([199.5 - 16, -0.5])
        ax2.set_ylim([99.5 - 8, -0.5])
        ax1.set_xticks(np.arange(0, 10001, 2000))
        ax1.set_xticklabels(np.arange(0, 11, 2, dtype=int), fontdict=font_ticks)
        ax1.set_yticks(np.arange(0, 201 - 16, 60))
        ax1.set_yticklabels(np.arange(0, 201 - 16, 60), fontdict=font_ticks)
        ax2.set_yticks(np.arange(0, 101 - 8, 30))
        ax2.set_yticklabels(np.arange(0, 101 - 8, 30), fontdict=font_ticks)
        filename = 'raster_STmap.png'
    plt.tight_layout()
    plt.savefig(f'{RES_PATH}\\{case_name}\\{filename}', transparent=True)
    plt.show()


def draw_SMC(id_SMC=60):
    fig, ax = plt.subplots(7, 1, sharex=True)
    id_CMN = id_SMC // 2
    for id_IPAN in range(max(id_CMN - 4, 0), id_CMN + 3):
        ax[0].plot(time_Tavg, DSTND[id_IPAN], label=id_IPAN)
        ax[1].plot(t, vIPAN[id_IPAN], label=id_IPAN)
    ax[2].plot(t, vECMN[id_CMN], label='ECMN')
    ax[2].plot(t, vICMN[id_CMN], label='ICMN')
    ax[3].plot(t, vSMC[id_SMC], label=id_SMC)
    # ax[3].plot(t, vICC[id_SMC], label='ICC')
    ax[4].plot(time_Tavg, TSMC[id_SMC], label=id_SMC)
    ax[4].plot(t, net['mSMC'].T[id_SMC], label=id_SMC)
    # ax[5].plot(t, net['mSMC'].I_CaL[id_SMC] / nA, label='CaL')
    ax[5].plot(t, net['mSMC'].I_BK[id_SMC] / nA, label='BK')
    # ax[5].plot(t, net['mSMC'].I_Kr[id_SMC] / nA, label='Kr')
    ax[5].plot(t, net['mSMC'].I_EJP[id_SMC] / nA, label='EJP')
    ax[5].plot(t, net['mSMC'].I_IJP[id_SMC] / nA, label='IJP')
    # ax[5].plot(t, net['mSMC'].I_GJ_SMC_ICC[id_SMC] / nA, label='ICC')
    ax[5].plot(t, net['mSMC'].I_GJ_SMC_SMC[id_SMC] / nA, label='SMC')
    # ax2[6].plot(t, net['mICC'].cGMP[id_SMC])
    # ax2[6].plot(t, net['mICC'].h_ANO1[id_SMC])
    ax[6].plot(t, net['mSMC'].Ca_i[id_SMC] / uM)
    ax[6].plot(t, net['mSMC'].Ca_i[id_SMC] / net['mSMC'].cGMP[id_SMC] / uM)
    ax[1].legend(loc='best', ncol=3)
    ax[2].legend(loc='best')
    ax[3].legend(loc='best')
    ax[5].legend(loc='best', ncol=5)
    # plt.xlim([0,3000])
    plt.show()


def SMC_details_paper(id_SMC, bars=True):
    """
    Plot membrane potential, intracellular calcium concentration, and tension of an SMC.
    :param id_SMC: SIP_ID
    :param bars: whether to display scale bar
    """
    global net, vICC, vSMC
    plt.rc('font', size=20, family='Arial')
    plt.rc('lines', lw=2)
    fig, ax = plt.subplots(
        3, 1, sharex=True, sharey='row', layout='constrained', figsize=(9, 8))
    ax[0].plot(t, vSMC[id_SMC])
    ax[0].axhline(-74, color='gray', ls='--')
    ax[1].plot(t, net['mSMC'].Ca_i[id_SMC] / uM)
    ax[1].plot(t, net['mSMC'].Ca_i[id_SMC] / net['mSMC'].cGMP[id_SMC] / uM)
    ax[1].axhline(0, color='gray', ls='--')
    ax[2].plot(t, net['mSMC'].T[id_SMC])
    ax[2].axhline(0, color='gray', ls='--')
    if bars:
        ax[0].plot([20600, 20600], [-50, -30], 'k')
        ax[1].plot([20600, 20600], [1, 2], 'k')
        ax[2].plot([18600, 20600], [8e-6, 8e-6], 'k')
        ax[2].plot([18600, 18600], [8e-6, 13e-6], 'k')
    ax[0].set_ylim([-85, -17])
    ax[1].set_ylim([0, 2.7])
    ax[2].set_ylim([0, 22e-6])
    for cur_ax in ax.flatten():
        for spine in cur_ax.spines.values():
            spine.set_visible(False)
        cur_ax.set_xticks([])
        cur_ax.set_xticklabels([])
        cur_ax.set_yticks([])
        cur_ax.set_yticklabels([])
    fig.savefig(f'{case_path}/SMC-details-#{id_SMC}.png', dpi=300)
    plt.show()


def SMC_network_paper(id_SMC):
    """Show how an SMC is activated in a network"""
    from tqdm import trange
    folder = f'{RES_PATH}\\{case_name}\\SMC{id_SMC}'
    os.makedirs(folder, exist_ok=True)
    plt.rc('font', **font_ticks)
    Dy = 250
    id_CMN = id_SMC // 2
    linelength = 80
    event_linewidth = 1
    lw = 2
    print('Processing Oral IPAN and DIN')
    for id_IPAN in trange(max(id_CMN - 4, 0), id_CMN):
        fig = plt.figure(figsize=(7.5, 2))
        plt.plot(time_Tavg, DSTND[id_IPAN] * 400 - 50, color='C4', lw=lw)
        print(id_IPAN, np.ptp(DSTND[id_IPAN]))
        plt.eventplot(t_spikeIPAN[id_IPAN], lineoffsets=Dy, colors='C4',
                      linelengths=linelength, linewidths=event_linewidth)
        plt.eventplot(t_spikeDIN[id_IPAN], lineoffsets=Dy * 2, colors='C3',
                      linelengths=linelength, linewidths=event_linewidth)
        plt.xlim([0, 25000])
        plt.ylim(-250, 550)
        # plt.show()
        for name, spine in plt.gca().spines.items():
            spine.set_visible(False)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        fig.savefig(f'{folder}\\IPAN{id_IPAN}', transparent=True, dpi=300)
        fig.savefig(f'{folder}\\IPAN{id_IPAN}.svg', transparent=True)
        plt.close()
        # plt.show()
    print('Processing Anal IPAN and AIN')
    for id_IPAN in trange(id_CMN + 1, id_CMN + 3):
        fig = plt.figure(figsize=(7.5, 2))
        plt.plot(time_Tavg, DSTND[id_IPAN] * 400 - 50, color='C4', lw=lw)
        plt.eventplot(t_spikeIPAN[id_IPAN], lineoffsets=Dy, colors='C4',
                      linelengths=linelength, linewidths=event_linewidth)
        plt.eventplot(t_spikeAIN[id_IPAN], lineoffsets=Dy * 2, colors='C2',
                      linelengths=linelength, linewidths=event_linewidth)
        plt.xlim([0, 25000])
        plt.ylim(-250, 550)
        # plt.show()
        for name, spine in plt.gca().spines.items():
            spine.set_visible(False)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        fig.savefig(f'{folder}\\IPAN{id_IPAN}', transparent=True, dpi=300)
        fig.savefig(f'{folder}\\IPAN{id_IPAN}.svg', transparent=True)
        plt.close()
    print('Processing current SMC')
    fig = plt.figure(figsize=(7.5, 3))
    plt.eventplot(t_spikeECMN[id_CMN], lineoffsets=0, color='C0',
                  linelengths=linelength, linewidths=event_linewidth)
    plt.eventplot(t_spikeICMN[id_CMN], lineoffsets=Dy, color='C1',
                  linelengths=linelength, linewidths=event_linewidth)

    # plt.plot(t, vICC[id_SMC] * 5 + Dy * 3.5, color='k', lw=lw, label='ICC')
    plt.plot(t, vSMC[id_SMC] * 5 + Dy * 3.5, color='k', lw=lw, label=id_SMC)
    print(np.ptp(vSMC[id_SMC]))
    plt.plot(t, net['mSMC'].Ca_i[id_SMC] / uM * 300 + Dy * 3.5, color='C1', lw=lw, label=id_SMC)
    print(np.ptp(net['mSMC'].Ca_i[id_SMC] / uM))
    plt.plot(time_Tavg, TSMC[id_SMC] * 25 + Dy * 5, lw=lw, label=id_SMC)
    print(np.ptp(TSMC[id_SMC]))
    plt.xlim([0, 25000])
    # plt.show()
    plt.xticks([], [])
    plt.yticks([], [])
    for name, spine in plt.gca().spines.items():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(f'{folder}\\SMC.png', dpi=300, transparent=True)
    fig.savefig(f'{folder}\\SMC.svg', transparent=True)
    plt.close()
    print('Draw Time axis')
    fig = plt.figure(figsize=(7.5, 2))
    plt.xlim([0, 25])
    plt.xticks([0, 5, 10, 15, 20, 25], ['0', '5', '10', '15', '20', '25'])
    plt.xlabel('Time (s)', fontdict=font_label)
    for name, spine in plt.gca().spines.items():
        if name != 'bottom':
            spine.set_visible(False)
    plt.yticks([], [])
    plt.tight_layout()
    fig.savefig(f'{folder}\\axis.png', dpi=300, transparent=True)
    fig.savefig(f'{folder}\\axis.svg', transparent=True)
    plt.show()


def draw_SMCs():
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t, vICC[100:114].T)
    ax[1].plot(t, vSMC[100:114].T)
    ax[2].plot(time_Tavg, TSMC[100:114].T)
    plt.show()


def drawIPAN(ind=0):
    fig, ax = plt.subplots(3, 1, sharex=True)
    for ind, ls in zip([39, 40], ['-', '--']):
        ax[0].plot(t, DSTND[ind], ls=ls)
        ax[1].plot(t, vIPAN[ind], ls=ls)
        ax[2].plot(t, net['mIPAN'].I_Na[ind] / nA, 'C0', label='Na', ls=ls)
        ax[2].plot(t, net['mIPAN'].I_K[ind] / nA, 'C1', label='K', ls=ls)
        ax[2].plot(t, net['mIPAN'].I_KA[ind] / nA, 'C2', label='KA', ls=ls)
        ax[2].plot(t, net['mIPAN'].I_SAC[ind] / nA, 'C3', label='SAC', ls=ls)
        ax[2].plot(t, net['mIPAN'].I_EPSP[ind] / nA, 'C4', label='EPSP', ls=ls)
        ax[2].plot(t, net['mIPAN'].I_AH[ind] / nA, 'C5', label='AH', ls=ls)
        ax[2].legend(loc='best')
    plt.show()


def calcium_map(ax=None, vline=None):
    """
    :param ax: external, optional
    :param vline: same-x line (mm)
    """
    from utils.id2x import x2ringID
    if ax:
        external_ax = True
    else:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
        external_ax = False
    cai_map = np.zeros((200, 25000))
    cai_map[8:-8] = net['mSMC'].Ca_i[:, ::2] / uM
    im = ax.imshow(cai_map.T, aspect='auto', vmin=0, vmax=1.5, cmap='viridis')
    cb = plt.colorbar(im)
    cb.set_label(r'$\mathrm{[Ca^{2+}]_i}$ (μM)', fontdict=font_label)
    cb.set_ticks(ticks=[0, 0.5, 1.0, 1.5])
    xticklabels = np.array([10, 20, 30])
    xticks = x2ringID(xticklabels)
    yticklabels = np.arange(0, 26, 5)
    yticks = yticklabels * 1000
    if vline:
        ax.axvline(vline, color='w', ls='--')
    if external_ax:
        return xticks, xticklabels, yticks, yticklabels
    plt.xticks(xticks, xticklabels)
    plt.yticks(yticks, yticklabels)
    plt.xlabel('x (mm)', fontdict=font_label)
    plt.ylabel('Time (s)', fontdict=font_label)
    plt.show()


def plot_activation_window(waveID_global, SIPIDs=None, ax=None, use_reverse_waveID=False):
    """
    For each given SIPID, find its activation window
    (duration of the window phase from the ending of ICMN inhibition to the ending of ECMN excitation)
    along the given waveID.
    Users always specify the global waveIDs, but the program needs the local ones (for an SMC).
    For later waves with large global waveIDs, users can let the program use reverse waveIDs to avoid
    unexpected errors.
    :param waveID_global: self-explanatory
    :param SIPIDs: list of SIPIDs
    :param ax: external axes
    :param use_reverse_waveID: whether map global waveIDs into local reverse waveIDs.
    """
    from utils.mathfunc import find_clusters
    from scipy.signal import find_peaks
    with open(f'{case_path}/waveID_global2local.pkl', 'rb') as pf:
        waveID_global2local = pickle.load(pf)
    if ax:
        external_ax = True
    else:
        plt.rc('font', **font_ticks)
        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
        external_ax = False
    if SIPIDs is None:
        SIPIDs = np.arange(0, muscle_pop)
    windows = np.zeros_like(SIPIDs, dtype=float)
    for i, SIPID in enumerate(SIPIDs):
        # i, SIPID = 1, 13
        ringID = SIPID + 8
        if waveID_global not in waveID_global2local[ringID]:
            print(f'ring #{ringID} is not involved in this wave')
            windows[i] = np.nan
            continue
        waveID_local = waveID_global2local[ringID][waveID_global]
        peak_ids, _ = find_peaks(TSMC[SIPID], prominence=1)
        neuronID = SIPID // 2
        t_spikeECMN_cur, t_spikeICMN_cur = t_spikeECMN[neuronID], t_spikeICMN[neuronID]
        ECMN_starts, ECMN_ends = find_clusters(t_spikeECMN_cur, 500)
        if i > 0:
            # ICMN #0 never works
            ICMN_starts, ICMN_ends = find_clusters(t_spikeICMN_cur, 500)
        else:
            ICMN_starts = np.zeros_like(ECMN_starts)
            ICMN_ends = ICMN_starts
        if use_reverse_waveID:
            waveID_local_reverse = -len(waveID_global2local[ringID]) + waveID_local
            print(f'ring #{ringID}, reverseID is {waveID_local_reverse}')
            waveID = waveID_local_reverse
        else:
            if len(peak_ids) <= waveID_local:
                print(f'SMC #{SIPID} has no such active force peaks')
                windows[i] = np.nan
                continue
            if len(ECMN_ends) <= waveID_local:
                print(f'ECMN #{neuronID} has no such spike bursts.')
                windows[i] = np.nan
                continue
            if len(ICMN_ends) <= waveID_local:
                print(f'ICMN #{neuronID} has no such spike bursts.')
                windows[i] = np.nan
                continue
            waveID = waveID_local
        if len(ECMN_starts) == len(ICMN_starts) == len(peak_ids):
            t_ECMN_start = t_spikeECMN_cur[ECMN_starts[waveID]]
            t_ECMN_end = t_spikeECMN_cur[ECMN_ends[waveID] - 1]
            t_ICMN_end = t_spikeICMN_cur[ICMN_ends[waveID] - 1]
        else:
            # When clusters with less population exist, cluster count can be larger than wave count.
            # Then we choose the cluster by the time points of tension peaks.
            t_tension_peak = peak_ids[waveID]  # ms
            nearest_ECMN_cluster = np.argmin(np.abs(t_tension_peak - t_spikeECMN_cur[ECMN_ends - 1]))
            t_ECMN_start = t_spikeECMN_cur[ECMN_starts[nearest_ECMN_cluster]]
            t_ECMN_end = t_spikeECMN_cur[ECMN_ends[nearest_ECMN_cluster] - 1]
            if i > 0:
                if len(ECMN_starts) == len(ICMN_starts):
                    nearest_ICMN_cluster = nearest_ECMN_cluster
                else:
                    nearest_ICMN_cluster = np.argmin(np.abs(t_ECMN_start - t_spikeICMN_cur[ICMN_starts]))
                t_ICMN_end = t_spikeICMN_cur[ICMN_ends[nearest_ICMN_cluster] - 1]
            else:
                t_ICMN_end = 0
        if t_ICMN_end > t_ECMN_start:
            windows[i] = t_ECMN_end - t_ICMN_end
        else:
            print(f'For SMC #{SIPID}, t_ICMN_end <= t_ECMN_start')
            windows[i] = t_ECMN_end - t_ECMN_start
    ringIDs = np.arange(8, 200 - 8)
    ax.plot(ringIDs, windows / 1000)
    if not external_ax:
        plt.show()


def draft():
    plt.plot(t, net['mSN'].zEPSP[:92].T)
    plt.show()


if __name__ == '__main__':
    # visualize_dynamics(draw_ICC=False)
    # draw_ICC()
    # raster_plot_tension(enlarge=False, colorbar=True)
    # drawIPAN(ind=29)
    # draw_SMC(id_SMC=59)
    # SMC_details_paper(id_SMC=24, bars=False)
    SMC_network_paper(id_SMC=59)
    # draw_SMCs()
    # calcium_map()
    # plot_activation_window(waveID_global=5, use_reverse_waveID=True)
    # draft()
    # pass
