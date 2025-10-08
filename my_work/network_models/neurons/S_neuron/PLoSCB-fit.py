"""
Adapted from the following article:
    Zarei Eskikand, P., Koussoulas, K., Gwynne, R. M. & Bornstein, J. C.
    Computational simulations and Ca2+ imaging reveal that slow synaptic
    depolarizations (slow EPSPs) inhibit fast EPSP evoked action potentials
    for most of their time course in enteric neurons. PLOS Comput. Biol. 18, e1009717 (2022).
"""
from brian2 import *
from brian2modelfitting import *
from collections import OrderedDict
from brian2 import codegen
from socket import gethostname
import efel
from utils.fitting import MyFeatureMetric

if gethostname() == 'gpu-server':
    codegen.cpp_prefs._compiler_supports_c99 = True
prop_units = OrderedDict(
    G_Nav13=siemens / cm2, G_Kdr=siemens / cm2,
    G_KM=siemens / cm2, G_leak=siemens / cm2
)
params_all = []
error_all = []


def callback(params, errors, best_params, best_error, index):
    global params_all, error_all
    params_all.extend(params)
    error_all.extend(errors)
    str_best = ''
    for name, unit in prop_units.items():
        val = best_params[name]
        str_best += f'{name}={val / unit: .4g}*siemens/cm2,\t'
    print(f'Round {index}: Error: {best_error:.4g}. '
          f'Best params: {str_best.strip()[:-1]}')


data_exp = load('ploscb-fig1.npy').T * mV
data_exp = data_exp[:10]
I_stim = zeros(data_exp.shape[1]) * nA
# I_stim[10000:60000] = 0.1 * nA
I_stim = TimedArray(I_stim, dt=0.01 * ms)
have_KM = zeros_like(data_exp)
have_KM[0] = 1
init = {
    'v': -55 * mV,
    'm_Nav13': 0.01014601, 'h_Nav13': 0.5,
    'n_Kdr': 0.09975049,
    'm1_KM': 0.12986263, 'm2_KM': 0.12986263
}

temp = 310.15
Cm = 1 * uF / cm2
ENa = 55 * mV
EK = -85 * mV
EL = -55 * mV
q_Nav13 = 2 ** ((temp - (24 + 273.15)) / 10)
q_KM = 5 ** ((temp - (22 + 273.15)) / 10)
eqs = '''
# Nav13
I_Nav13 = G_Nav13*m_Nav13**3*h_Nav13*(ENa-v) : amp/meter**2
dm_Nav13/dt = (m_inf_Nav13-m_Nav13)/tau_m_Nav13 : 1
alpha_m_Nav13 = (4.5/exprel(-(v/mV+29.5)/4.5) * 0.4/ms) * int(abs(v/mV+29.5)>1e-6) +
                1.8/ms * int(abs(v/mV+29.5)<=1e-6) : hertz
beta_m_Nav13 = (4.5/exprel((v/mV+29.5)/4.5) * 0.135/ms) * int(abs(v/mV+29.5)>1e-6) +
                0.6075/ms * int(abs(v/mV+29.5)<=1e-6) : hertz
tau_m_Nav13 = clip(1/((alpha_m_Nav13+beta_m_Nav13)*q_Nav13),0.02*ms,inf*ms) : second
m_inf_Nav13 = alpha_m_Nav13/(alpha_m_Nav13+beta_m_Nav13) : 1
dh_Nav13/dt = (h_inf_Nav13-h_Nav13)/tau_h_Nav13 : 1
alpha_h_Nav13 = 1.5/exprel(-(v/mV+30)/1.5) * 0.03/ms * int(abs(v/mV+30)>1e-6) +
                0.045/ms * int(abs(v/mV+30)<=1e-6) : hertz
beta_h_Nav13 = 1.5/exprel((v/mV+30)/1.5) * 0.01 / ms * int(abs(v/mV+30)>1e-6) +
                0.015/ms * int(abs(v/mV+30)<=1e-6): hertz
tau_h_Nav13 = clip(1/((alpha_h_Nav13+beta_h_Nav13)*q_Nav13),0.5*ms,inf*ms) : second
h_inf_Nav13 = 1/(1+exp((v/mV+55)/4)) : 1

# Kdr
I_Kdr = G_Kdr*n_Kdr**4*(EK-v) : amp/meter**2
dn_Kdr/dt = (n_inf_Kdr-n_Kdr)/tau_n_Kdr : 1
tau_n_Kdr = (0.25+4.35*exp(-sign(v+10*mV)*(v/mV+70)/15))*ms : second
n_inf_Kdr = 1/(1+exp(-(v/mV)/25)) : 1

# KM
I_KM = G_KM * m1_KM**3 * m2_KM * (EK-v) : amp/meter**2
# z1*gamma*frt == 16.2463, z2*gamma*frt == 51.64
dm1_KM/dt = (m_inf_KM-m1_KM)/tau_m1_KM : 1
dm2_KM/dt = (m_inf_KM-m1_KM)/tau_m2_KM : 1
tau_m1_KM = (176.1*0.5/cosh((v/mV+20)*16.2463/temp)+20.7)/q_KM*ms : second
tau_m2_KM = (1473*0.5/cosh((v/mV+20)*51.64/temp)+149)/q_KM*ms : second
m_inf_KM = 1/(1+exp(-(v/mV+20)/18.4)) : 1

# leak channel
I_leak = G_leak*(EL-v) : amp/meter**2

dv/dt = (I_Nav13+I_Kdr+I_KM*have_KM+I_leak+I_stim(t)/(3848.451*um2))/Cm : volt

G_Nav13 : siemens / meter**2 (constant)
G_Kdr : siemens / meter**2 (constant)
G_KM : siemens / meter**2 (constant)
G_leak : siemens / meter**2 (constant)
'''

fitter = TraceFitter(
    model=eqs,
    input_var='have_KM',
    output_var='v',
    input=have_KM,
    output=data_exp,
    dt=0.01 * ms,
    n_samples=150,
    method='exponential_euler',
    param_init=init
)
efel.set_setting('Threshold', 10)
weight_feat = {
    'mean_frequency': 1,
    'mean_AP_amplitude': 1,
    'spike_count': 1.5,
    'voltage_base': 1,
    'voltage_after_stim': 0.5,
    'min_AHP_values': 0.6
}
res, error = fitter.fit(
    n_rounds=100,
    optimizer=NevergradOptimizer(use_nevergrad_recommendation=True),
    metric=MyFeatureMetric(
        stim_times=[(100 * ms, 600 * ms)],
        feat_list=weight_feat.keys(),
        weights=weight_feat,
        nan_replace=1000
    ),
    callback=callback,
    G_Nav13=[0.05 * siemens / cm2, 0.2 * siemens / cm2],
    G_Kdr=[0.001 * siemens / cm2, 0.4 * siemens / cm2],
    G_KM=[0.01 * siemens / cm2, 0.12 * siemens / cm2],
    G_leak=[5e-6 * siemens / cm2, 4e-4 * siemens / cm2]
)
trace = fitter.generate_traces()
ts = arange(data_exp.shape[1]) * 0.01
fig, ax = subplots(2, 1, sharex=True)
for i in range(2):
    ax[i].plot(ts, data_exp[i] / mV)
    ax[i].plot(ts, trace[i] / mV)
show()
