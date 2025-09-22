from brian2 import *
from brian2modelfitting import *
from utils.fitting import CaLMetric
import pickle
from collections import OrderedDict
import pandas as pd

prop_units = OrderedDict(
    # CaL
    k_d_CaL=mV,
    V_half_f_CaL=mV, k_f_CaL=mV,
    h_Ca_CaL=uM, s_Ca_CaL=uM,
    tau_d_CaL=ms, tau_f_CaL=ms, tau_fCa_CaL=ms,
    G_CaL=nS,
    # CaT
    V_half_d_CaT=mV, k_d_CaT=mV,
    V_half_f_CaT=mV, k_f_CaT=mV,
    tau_d_CaT=ms, tau_f_CaT_C1=ms, tau_f_CaT_C2=1 / mV,
    G_CaT=nS,
    # Kr
    V_half_x_r1=mV, k_x_r1=mV,
    V_half_x_r2=mV, k_x_r2=mV, scale_x_r2=1,
    tau_x_r1=ms, tau_x_r2_C1=ms, tau_x_r2_C2=ms,
    tau_x_r2_C3=mV, tau_x_r2_C4=mV,
    G_Kr=nS,
    # Ka
    V_half_x_a1=mV, k_x_a1=mV,
    V_half_x_a2=mV, k_x_a2=mV, scale_x_a2=1,
    tau_x_a1_C1=ms, tau_x_a1_C2=ms, tau_x_a1_C3=1,
    tau_x_a1_C4=mV, tau_x_a1_C5=mV,
    tau_x_a2=ms,
    G_Ka=nS,
    # BK
    h_BK=1, K_BK=mV,
    G_BK=nS,
    # Kb
    G_Kb=nS,
    # Na
    V_half_d_Na=mV, k_d_Na=mV,
    V_half_f_Na=mV, k_f_Na=mV,
    tau_d_Na_C1=1 / mV * ms, tau_d_Na_C2=ms,
    tau_f_Na_C1=1 / mV * ms, tau_f_Na_C2=ms,
    G_Na=nS,
    # NCX
    gamma_NCX=1, K_mNa_i_NCX=mM, K_mCa_NCX=mM,
    k_sat_NCX=1, P_NCX=amp,
    # GJ ICC-SMC
    G_GJ_SMC_ICC=nS,
)
params_all = []
error_all = []


def callback(params, errors, best_params, best_error, index):
    global params_all, error_all
    params_all.extend(params)
    error_all.extend(errors / mV ** 2)
    str_best = ''
    for name, unit in prop_units.items():
        val = best_params[name]
        if isinstance(unit, int):  # unit == 1
            str_best += f'{name}={val: .4g},\t'
        else:
            str_best += f'{name}={val / unit: .4g}*{unit},\t'
    print(f'Round {index}: Error: {best_error / mV ** 2:.4g}*mV2. '
          f'Best params: {str_best.strip()[:-1]}')


with open('2003-SMC-ICC-superimposed.pkl', 'rb') as f:
    data = pickle.load(f)
    t_cycle_ICC = data['t_cycle_ICC'] * ms
    v_ICC = data['v_ICC']
    v_SMC_exper = data['v_SMC']
v_ICC = TimedArray(r_[ones(10000) * -76, v_ICC] * mV, dt=defaultclock.dt)

v_SMC_exper = r_[ones(10000) * -76, v_SMC_exper] * mV
v_SMC_exper = tile(v_SMC_exper, (2, 1))
have_CaL = zeros_like(v_SMC_exper)
have_CaL[1] = 1
t_wt = zeros_like(v_SMC_exper)
t_wt[0, 10000:] = 0.5
t_wt[0, 16000:29000] = 0.8
t_wt[0, 17000:20000] = 1
t_wt[1, 10000:] = 0.5
t_wt[1, 16000:29000] = 1
t_wt[1, 17000:19500] = 0

init = {
    # state var
    'v': -76 * mV, 'Ca_i': 8e-5 * mM,
    'd_CaT': 0.02030952, 'f_CaT': 0.99955,
    'x_r2': 0.81123808, 'x_a1': 0.00417406, 'x_a2': 0.38558536,
    'd_Na': 0.00866652, 'f_Na': 0.06008665,
    'CaM': 0.9285 * uM, 'MLCK': 9.6506 * uM,
    'Ca2CaM': 0.0015 * uM, 'Ca4CaM': 0 * uM,
    'CaM_MLCK': 0.3332 * uM, 'Ca2CaM_MLCK': 0.2713 * uM, 'Ca4CaM_MLCK': 0.0130 * uM,
    'CaM_BP': 2.8207 * uM, 'BP': 15.1793 * uM
}
R = 8.3144 * joule / (mole * kelvin)
F = 96486 * coulomb / mole
temp = 310 * kelvin
temp_exp = 297 * kelvin
Q10_Ca = 2.1
Q10_K = 1.365
Q10_Na = 2.45
Q_BK = 1.1 / kelvin * nS
T_correct_Ca = Q10_Ca ** ((temp - temp_exp) / (10 * kelvin))
T_correct_K = Q10_K ** ((temp - temp_exp) / (10 * kelvin))
T_correct_Na = Q10_Na ** ((temp - temp_exp) / (10 * kelvin))
T_correct_BK = Q_BK * (temp - temp_exp)
Ca_o = 2.5 * mM
Na_o = 137 * mM
Na_i = 10 * mM
K_o = 5.9 * mM
K_i = 164 * mM
Cm = 77 * pF
tau_x_a1_C6 = 2
Ca_set_BK = 0.001 * mM
Vc = 3500 * um3
RToF = R * temp / F
EK = RToF * log(K_o / K_i)
ENa = RToF * log(Na_o / Na_i)
V_half_d_CaL = 17 * mV
# MLCK activation
kf1 = 12 / (uM ** 2 * second)
kr1 = 12 / second
kf2 = 480 / (uM ** 2 * second)
kr2 = 1200 / second
kf3 = 5 / (uM * second)
kr3 = 135 / second
kf4 = 840 / (uM * second)
kr4 = 45.4 / second
kf5 = 28 / (uM * second)
kr5 = 0.0308 / second
kf6 = 120 / (uM ** 2 * second)
kr6 = 4 / second
kf7 = 7.5 / (uM ** 2 * second)
kr7 = 3.75 / second
kf8 = 5 / (uM * second)
kr8 = 25 / second
kf9 = 7.6 / (uM ** 2 * second)
kr9 = 22.8 / (uM * second)

eqs = """
ECa = RToF * 0.5 * log(Ca_o / Ca_i) : volt
# CaL
I_CaL = G_CaL*d_CaL*f_CaL*fCa_CaL*(ECa-v)*have_CaL : amp
dd_CaL/dt = (d_CaL_inf-d_CaL)/(tau_d_CaL*T_correct_Ca) : 1
d_CaL_inf = 1/(1+exp(-(v+V_half_d_CaL)/k_d_CaL)) : 1
df_CaL/dt = (f_CaL_inf-f_CaL)/(tau_f_CaL*T_correct_Ca) : 1
f_CaL_inf = 1/(1+exp((v+V_half_f_CaL)/k_f_CaL)) : 1
dfCa_CaL/dt = (fCa_CaL_inf-fCa_CaL)/(tau_fCa_CaL*T_correct_Ca) : 1
fCa_CaL_inf = 1-1/(1+exp(-(Ca_i-h_Ca_CaL)/(s_Ca_CaL))) : 1
# CaT
I_CaT = G_CaT*d_CaT*f_CaT*(ECa-v) : amp
dd_CaT/dt = (d_CaT_inf-d_CaT)/(tau_d_CaT*T_correct_Ca) : 1
d_CaT_inf = 1/(1+exp(-(v+V_half_d_CaT)/k_d_CaT)) : 1
df_CaT/dt = (f_CaT_inf-f_CaT)/(tau_f_CaT*T_correct_Ca) : 1
f_CaT_inf = 1/(1+exp((v+V_half_f_CaT)/k_f_CaT)) : 1
tau_f_CaT = tau_f_CaT_C1*exp(tau_f_CaT_C2*v) : second
# Kr
I_Kr = G_Kr*x_r1*x_r2*(EK-v) : amp
dx_r1/dt = (x_r1_inf-x_r1)/(tau_x_r1*T_correct_K) : 1
x_r1_inf = 1/(1+exp(-(v+V_half_x_r1)/k_x_r1)) : 1
dx_r2/dt = (x_r2_inf-x_r2)/(tau_x_r2*T_correct_K) : 1
x_r2_inf = scale_x_r2/(1+exp((v+V_half_x_r2)/k_x_r2))+1-scale_x_r2 : 1
tau_x_r2 = -tau_x_r2_C1+tau_x_r2_C2*exp((v+tau_x_r2_C3)/tau_x_r2_C4) : second
# Ka
I_Ka = G_Ka*x_a1*x_a2*(EK-v) : amp
dx_a1/dt = (x_a1_inf-x_a1)/(tau_x_a1*T_correct_K) : 1
x_a1_inf = 1/(1+exp(-(v+V_half_x_a1)/k_x_a1)) : 1
tau_x_a1 = tau_x_a1_C1+tau_x_a1_C2*exp(tau_x_a1_C3*((v+tau_x_a1_C4)/tau_x_a1_C5)**tau_x_a1_C6) : second
dx_a2/dt = (x_a2_inf-x_a2)/(tau_x_a2*T_correct_K) : 1
x_a2_inf = scale_x_a2/(1+exp(-(v+V_half_x_a2)/k_x_a2))+1-scale_x_a2 : 1
# BK
I_BK = (G_BK+T_correct_BK)*P0_BK*(EK-v) : amp
P0_BK = 1/(1+exp(v/K_BK-h_BK*log(Ca_i/Ca_set_BK))) : 1
# Kb
I_Kb = G_Kb*(EK-v) : amp
# Na
I_Na = G_Na*d_Na*f_Na*(ENa-v) : amp
dd_Na/dt = (d_Na_inf-d_Na)/(tau_d_Na*T_correct_Na) : 1
d_Na_inf = 1/(1+exp(-(v+V_half_d_Na)/k_d_Na)) : 1
tau_d_Na = tau_d_Na_C1*v+tau_d_Na_C2 : second
df_Na/dt = (f_Na_inf-f_Na)/(tau_f_Na*T_correct_Na) : 1
f_Na_inf = 1/(1+exp((v+V_half_f_Na)/k_f_Na)) : 1
tau_f_Na = tau_f_Na_C1*v+tau_f_Na_C2 : second
# NCX
nom_NCX = 2.5*exp((gamma_NCX-1)*v/RToF)*Na_o**3*Ca_i - exp(gamma_NCX*v/RToF)*Na_i**3*Ca_o: mM**4
denom_NCX = (K_mNa_i_NCX**3+Na_o**3)*(K_mCa_NCX+Ca_o)*(1+k_sat_NCX*exp((gamma_NCX-1)*v/RToF)) : mM**4
I_NCX = nom_NCX / denom_NCX * P_NCX : amp
# Ca2+ dynamics
dCa_i/dt = (I_CaL+I_CaT-2*I_NCX)/(2*F*Vc) : mM
# gap junction
I_GJ = (v_ICC(t)-v)*G_GJ_SMC_ICC : amp
# membrane
dv/dt = (I_CaL+I_CaT+I_Kr+I_Ka+I_BK+I_Kb+I_Na+I_GJ+I_NCX)/Cm : volt

# MLCK activation =============================
# items
f1 = kf1 * CaM          * Ca_i**2   : mM/second
r1 = kr1 * Ca2CaM                   : mM/second
f2 = kf2 * Ca2CaM       * Ca_i**2   : mM/second
r2 = kr2 * Ca4CaM                   : mM/second
f3 = kf3 * CaM          * MLCK      : mM/second
r3 = kr3 * CaM_MLCK                 : mM/second
f4 = kf4 * Ca2CaM       * MLCK      : mM/second
r4 = kr4 * Ca2CaM_MLCK              : mM/second
f5 = kf5 * Ca4CaM       * MLCK      : mM/second
r5 = kr5 * Ca4CaM_MLCK              : mM/second
f6 = kf6 * CaM_MLCK     * Ca_i**2   : mM/second
r6 = kr6 * Ca2CaM_MLCK              : mM/second
f7 = kf7 * Ca2CaM_MLCK  * Ca_i**2   : mM/second
r7 = kr7 * Ca4CaM_MLCK              : mM/second
f8 = kf8 * CaM * BP                 : mM/second
r8 = kr8 * CaM_BP                   : mM/second
f9 = kf9 * CaM_BP       * Ca_i**2   : mM/second
r9 = kr9 * Ca2CaM       * BP        : mM/second
# reactions
dCaM/dt = -f1+r1-f3+r3-f8+r8        : mM
dCa2CaM/dt = f1-r1-f2+r2-f4+r4+f9-r9: mM
dCa4CaM/dt = f2-r2-f5+r5            : mM
dMLCK/dt = -f3+r3-f4+r4-f5+r5       : mM
dCaM_MLCK/dt = f3-r3-f6+r6          : mM
dCa2CaM_MLCK/dt = f4-r4+f6-r6-f7+r7 : mM
dCa4CaM_MLCK/dt = f5-r5+f7-r7       : mM
dBP/dt = -f8+r8+f9-r9               : mM
dCaM_BP/dt = f8-r8-f9+r9            : mM

# CaL
k_d_CaL : volt (constant)
V_half_f_CaL : volt (constant)
k_f_CaL : volt (constant)
h_Ca_CaL : mM (constant)
s_Ca_CaL : mM (constant)
tau_d_CaL : second (constant)
tau_f_CaL : second (constant)
tau_fCa_CaL : second (constant)
G_CaL : siemens (constant)
# CaT
V_half_d_CaT: volt (constant)
k_d_CaT: volt (constant)
V_half_f_CaT: volt (constant)
k_f_CaT: volt (constant)
tau_d_CaT: second (constant)
tau_f_CaT_C1: second (constant)
tau_f_CaT_C2: 1/volt (constant)
G_CaT: siemens (constant)
# Kr
V_half_x_r1: volt (constant)
k_x_r1: volt (constant)
V_half_x_r2: volt (constant)
k_x_r2: volt (constant)
scale_x_r2: 1 (constant)
tau_x_r1: second (constant)
tau_x_r2_C1: second (constant)
tau_x_r2_C2: second (constant)
tau_x_r2_C3: volt (constant)
tau_x_r2_C4: volt (constant)
G_Kr: siemens (constant)
# Ka
V_half_x_a1: volt (constant)
k_x_a1: volt (constant)
V_half_x_a2: volt (constant)
k_x_a2: volt (constant)
scale_x_a2: 1 (constant)
tau_x_a1_C1: second (constant)
tau_x_a1_C2: second (constant)
tau_x_a1_C3: 1 (constant)
tau_x_a1_C4: volt (constant)
tau_x_a1_C5: volt (constant)
tau_x_a2: second (constant)
G_Ka: siemens (constant)
# BK
h_BK: 1 (constant)
K_BK: volt (constant)
G_BK: siemens (constant)
# Kb
G_Kb: siemens (constant)
# Na
V_half_d_Na: volt (constant)
k_d_Na: volt (constant)
V_half_f_Na: volt (constant)
k_f_Na: volt (constant)
tau_d_Na_C1: second / volt (constant)
tau_d_Na_C2: second (constant)
tau_f_Na_C1: second / volt (constant)
tau_f_Na_C2: second (constant)
G_Na: siemens (constant)
# NCX
gamma_NCX: 1 (constant)
K_mNa_i_NCX : mM (constant)
K_mCa_NCX : mM (constant)
k_sat_NCX : 1 (constant)
P_NCX : amp (constant)
# GJ ICC-SMC
G_GJ_SMC_ICC : siemens (constant)
"""
fitter = TraceFitter(
    model=eqs,
    input_var='have_CaL',
    output_var='v',
    input=have_CaL,
    output=v_SMC_exper,
    dt=0.1 * ms,
    n_samples=150,
    method='euler',
    param_init=init
)

res, error = fitter.fit(
    n_rounds=1000,
    optimizer=NevergradOptimizer(),
    metric=CaLMetric(nan_replace=1000, t_wt=t_wt),
    callback=callback,
    # CaL
    k_d_CaL=[1 * mV, 7 * mV],
    V_half_f_CaL=[60 * mV, 90 * mV], k_f_CaL=[6 * mV, 20 * mV],
    h_Ca_CaL=[0.1 * uM, 1 * uM], s_Ca_CaL=[0.01 * uM, 0.03 * uM],
    tau_d_CaL=[0.5 * ms, 1.5 * ms], tau_f_CaL=[30 * ms, 100 * ms], tau_fCa_CaL=[1. * ms, 6 * ms],
    G_CaL=[10 * nS, 90 * nS],
    # CaT
    V_half_d_CaT=[42 * mV, 52 * mV], k_d_CaT=[5 * mV, 17 * mV],
    V_half_f_CaT=[65 * mV, 75 * mV], k_f_CaT=[4 * mV, 8.3 * mV],
    tau_d_CaT=[2 * ms, 7 * ms], tau_f_CaT_C1=[25 * ms, 75 * ms], tau_f_CaT_C2=[1e-3 / mV, 5e-2 / mV],
    G_CaT=[0.01 * nS, 0.1 * nS],
    # Kr
    V_half_x_r1=[13 * mV, 34 * mV], k_x_r1=[1 * mV, 7 * mV],
    V_half_x_r2=[40 * mV, 80 * mV], k_x_r2=[2 * mV, 18 * mV], scale_x_r2=[0.5, 0.92],
    tau_x_r1=[65 * ms, 120 * ms], tau_x_r2_C1=[400 * ms, 850 * ms], tau_x_r2_C2=[1000 * ms, 2000 * ms],
    tau_x_r2_C3=[10 * mV, 50 * mV], tau_x_r2_C4=[60 * mV, 115 * mV],
    G_Kr=[1 * nS, 50 * nS],
    # Ka
    V_half_x_a1=[4 * mV, 22 * mV], k_x_a1=[4 * mV, 21 * mV],
    V_half_x_a2=[50 * mV, 78 * mV], k_x_a2=[4 * mV, 10 * mV], scale_x_a2=[0.85, 1],
    tau_x_a1_C1=[14 * ms, 35 * ms], tau_x_a1_C2=[100 * ms, 400 * ms], tau_x_a1_C3=[-1, -0.05],
    tau_x_a1_C4=[20 * mV, 55 * mV], tau_x_a1_C5=[10 * mV, 35 * mV],
    tau_x_a2=[30 * ms, 90 * ms],
    G_Ka=[0.5 * nS, 7 * nS],
    # BK
    h_BK=[3, 8], K_BK=[-70 * mV, -20 * mV],
    G_BK=[20 * nS, 85 * nS],
    # Kb
    G_Kb=[0.003 * nS, 0.017 * nS],
    # Na
    V_half_d_Na=[40 * mV, 80 * mV], k_d_Na=[3 * mV, 8.5 * mV],
    V_half_f_Na=[80 * mV, 100 * mV], k_f_Na=[2. * mV, 4 * mV],
    tau_d_Na_C1=[-0.026 / mV * ms, -0.005 / mV * ms], tau_d_Na_C2=[0.2 * ms, 0.6 * ms],
    tau_f_Na_C1=[-0.6 / mV * ms, -0.05 / mV * ms], tau_f_Na_C2=[1 * ms, 7 * ms],
    G_Na=[0.5 * nS, 5 * nS],
    # NCX
    gamma_NCX=[0.05, 0.95], K_mNa_i_NCX=[90 * mM, 200 * mM], K_mCa_NCX=[0.4 * mM, 2.5 * mM],
    k_sat_NCX=[0.1, 0.7], P_NCX=[4000 * pA, 8000 * pA],
    # GJ ICC-SMC
    G_GJ_SMC_ICC=[0.5 * nS, 20 * nS],
)
trace1 = fitter.generate_traces()
ts = arange(v_SMC_exper.shape[1]) * 0.1
plot(ts, v_SMC_exper[0] / mV, label='exp')
plot(ts, trace1[0] / mV, label='pred1')
plot(ts, trace1[1] / mV, label='pred2')
show()
history = pd.DataFrame(params_all)
history['error'] = error_all
for name, unit in prop_units.items():
    history[name] = history[name].apply(lambda x: x / unit)
history = history[['error', *prop_units.keys()]]
with open('CaL-opt-history.pkl', 'wb') as f1:
    pickle.dump(history, f1)
with open('CaL-params0.pkl', 'wb') as f:
    pickle.dump({'params': res}, f)
