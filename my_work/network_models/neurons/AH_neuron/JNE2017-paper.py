from brian2.only import *
import brian2.numpy_ as np
import matplotlib.pyplot as plt
from socket import gethostname

if gethostname() == 'LAPTOP-1QA0JPIO':
    path = 'F:/intestine_results'
else:
    path = '../../../results'
case_name =  'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
p_AH = dict(
    Cm=200 * pF,
    G_leak=0.06 * uS, G_Na=24 * uS, G_K=4 * uS, G_KA=9.54 * uS,
    EL=-17.0 * mV, EK=-72 * mV, ENa=55.0 * mV, E_KA=-75 * mV,
    # AH
    E_AH=-89 * mV,
    T_on=0.07 * second, T_off=0.5 * second,
    # SAC
    E_SAC=40 * mV, G_SAC=0.07 * uS,
)
# IPAN
eqs_AH = '''
# Na
I_Na= G_Na*m_Na**3*h_Na*(ENa-v) :  amp
dm_Na/dt = alpha_m_Na * (1 - m_Na) - beta_m_Na * m_Na : 1
dh_Na/dt = alpha_h_Na * (1 - h_Na) - beta_h_Na * h_Na : 1
alpha_m_Na = 0.38 * (v / mV + 29.7)/(1 - exp(-0.1 * (v / mV + 29.7))) / ms : Hz
beta_m_Na = 15.2 * exp(-0.0556 * (v / mV + 54.7)) / ms : Hz
alpha_h_Na = 0.266 * exp(-0.05*(v / mV + 48)) / ms : Hz
beta_h_Na = 3.8 / (1 + exp(-0.1 * (v / mV + 18))) / ms : Hz

# K
I_K = G_K*n_K**4*(EK-v) : amp
dn_K/dt = alpha_n_K * (1 - n_K) - beta_n_K * n_K : 1
alpha_n_K = (0.02 * (v / mV + 45.7)) / (1 - exp(-0.1 * (v / mV + 45.7))) / ms : Hz
beta_n_K = 0.25 * exp(-0.0125 * (v / mV + 55.7)) / ms : Hz

# KA
I_KA = G_KA*a_KA**3*b_KA*(E_KA-v) :  amp
da_KA/dt = (a_inf_KA-a_KA)/tau_a_KA : 1
a_inf_KA = ((0.0761*exp(0.0314/mV*(v+94.22*mV)))/(1+exp(0.0346/mV*(v+1.17*mV))))**(1/3) : 1
tau_a_KA = 0.3622*ms+1.158*ms/(1+exp(0.0497/mV*(v+55.96*mV))) : second
db_KA/dt = (b_inf_KA-b_KA)/tau_b_KA : 1
b_inf_KA = (1/(1+exp(0.0688/mV*(v+53.3*mV))))**4 : 1
tau_b_KA = 1.24*ms+2.678*ms/(1+exp(0.0624/mV*(v+50*mV))) : second

# AH
I_AH = gAH * (E_AH - v) : amp
dgAH/dt = (3 * nS - gAH) / T_off + zAH : siemens
dzAH/dt = -zAH/T_on : siemens * Hz

# leak
I_leak = G_leak*(EL-v) : amp

# SAC
I_SAC = G_SAC/(1+exp(-(DSTND-0.07)/0.003))*(E_SAC-v)*int(DSTND>0) : amp
# DSTND = distension(t,i) : 1
DSTND : 1
dv/dt = (I_Na+I_K+I_KA+I_AH+I_leak+I_SAC)/Cm : volt
'''

distension = np.arange(0.01, 0.16, 0.002)
N = distension.size
IPAN = NeuronGroup(
    N, eqs_AH, method='exponential_euler',
    threshold='v>10*mV',
    reset='zAH+=50*nS/second',
    refractory=1 * ms,
    namespace=p_AH, name='IPAN')
IPAN.v = -65 * mV
IPAN.m_Na = 'alpha_m_Na/(alpha_m_Na+beta_m_Na)'
IPAN.h_Na = 'alpha_h_Na/(alpha_h_Na+beta_h_Na)'
IPAN.n_K = 'alpha_n_K/(alpha_n_K+beta_n_K)'
IPAN.a_KA = 'a_inf_KA'
IPAN.b_KA = 'b_inf_KA'
IPAN.gAH = 0.08 * uS

mIPAN = StateMonitor(
    IPAN,
    ('v', 'I_leak', 'I_K', 'I_Na', 'I_KA', 'I_AH', 'I_SAC'
     ), record=True)
sIPAN = SpikeMonitor(IPAN, record=True)
# Run and Plot
IPAN.DSTND = distension
time_elapsed = 5 * second
run(time_elapsed)
# fig, ax2 = subplots(3, 1, sharex=True)
indices = list(range(2, N, 5))
print(indices)
time = mIPAN.t / ms
plt.rc('font', size=20, family='Arial')
font_label = {'size': 23, 'family': 'Arial'}
plt.rc('lines', lw=2)
fig, ax = plt.subplot_mosaic(
    [['V', 'FR'],
     ['I', 'FR']],
    layout='constrained',
    figsize=(20, 9),
    width_ratios=[2,1]
)
ind = 65
print(distension[ind])
ax['V'].plot(time, mIPAN.v[ind] / mV, 'k')
ax['V'].plot([1607, 1612], [-20, -20], 'k')
ax['V'].plot([1607, 1607], [-20, 0], 'k')
ax['V'].axhline(y=-65, color='gray', ls='--')
ax['V'].set_xlim([1595, 1630])
ax['I'].plot(time, mIPAN.I_Na[ind] / nA, label='NaHH')
ax['I'].plot(time, mIPAN.I_K[ind] / nA, label='KHH')
ax['I'].plot(time, mIPAN.I_KA[ind] / nA, label='KA')
ax['I'].plot(time, mIPAN.I_leak[ind] / nA, label='leak')
ax['I'].plot(time, mIPAN.I_AH[ind] / nA, label='AH')
ax['I'].plot(time, mIPAN.I_SAC[ind] / nA, label='SAC')
ax['I'].axhline(y=0, color='gray', ls='--')
ax['I'].plot([1604, 1604], [-90, -40], 'k')
ax['I'].set_xlim([1595, 1630])
ax['I'].legend(loc='lower center', prop=font_label, ncol=2)
inset_I = ax['I'].inset_axes(
    [1602, 20, 14, 110],
    transform=ax['I'].transData,
    xlim=(1597, 1601),
    ylim=(-15, 10)
)
inset_I.plot(time, mIPAN.I_Na[ind] / nA, label='NaHH')
inset_I.plot(time, mIPAN.I_K[ind] / nA, label='KHH')
inset_I.plot(time, mIPAN.I_KA[ind] / nA, label='KA')
inset_I.plot(time, mIPAN.I_leak[ind] / nA, label='leak')
inset_I.plot(time, mIPAN.I_AH[ind] / nA, label='AH')
inset_I.plot(time, mIPAN.I_SAC[ind] / nA, label='SAC')
inset_I.axhline(y=0, color='gray', ls='--')
inset_I.plot([1600.4, 1600.9], [-10, -10], 'k')
inset_I.plot([1600.4, 1600.4], [-10, -6], 'k')
ax['I'].indicate_inset_zoom(inset_I, edgecolor='k')
for cur_ax in [ax['V'], ax['I'], inset_I]:
    cur_ax.set_xticks([])
    cur_ax.set_xticklabels([])
    cur_ax.set_yticks([])
    cur_ax.set_yticklabels([])
    if cur_ax is not inset_I:
        for spine in cur_ax.spines.values():
            spine.set_visible(False)
ax['FR'].plot(distension * 100, sIPAN.count / time_elapsed, '.-', ms=20)
ax['FR'].set_xlabel('Strain (%)', fontdict=font_label)
ax['FR'].set_ylabel('Firing rate (Hz)', fontdict=font_label)
fig.savefig(f'{path}/{case_name}/IPAN-details.png', dpi=300)
plt.show()
