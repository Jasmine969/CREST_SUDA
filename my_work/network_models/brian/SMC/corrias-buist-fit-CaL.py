from brian2 import *
from brian2modelfitting import *
from utils.fitting import MSENaNMetric
import pickle
from collections import OrderedDict
import pandas as pd

prop_units = OrderedDict(
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
    # Ca2+ dynamics
    CRT_tot=mM, K_CRT_D=mM, CaM_tot=mM,
    K_CaM_D=mM ** 4
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


with open('../2003-SMC-ICC-superimposed.pkl', 'rb') as f:
    data = pickle.load(f)
    t_cycle_ICC = data['t_cycle_ICC'] * ms
    v_ICC = data['v_ICC']
    v_SMC_exper = data['v_SMC']
v_ICC = r_[ones(10000) * -76, v_ICC] * mV
t_wt = zeros_like(v_ICC)
t_wt[10000:] = 0.5
t_wt[16000:29000] = 0.8
t_wt[17000:20000] = 1
v_ICC = v_ICC[newaxis, :]
v_SMC_exper = r_[ones(10000) * -76, v_SMC_exper] * mV
v_SMC_exper = v_SMC_exper[newaxis, :]
G_GJ_SMC_ICC = 1.3 * nS
init = {
    # state var
    'v': -76 * mV, 'Ca_i': 8e-5 * mM,
    'd_CaT': 0.02030952, 'f_CaT': 0.99955,
    'x_r2': 0.81123808, 'x_a1': 0.00417406, 'x_a2': 0.38558536,
    'd_Na': 0.00866652, 'f_Na': 0.06008665
}
R = 8.3144 * joule / (mole * kelvin)
F = 96486 * coulomb / mole
temp = 310 * kelvin
temp_exp = 297 * kelvin,
Q10_Ca = 2.1
Q10_K = 1.365
Q10_Na = 2.45
Q_BK = 1.1 / kelvin * nS
correct_Ca = Q10_Ca ** ((temp - temp_exp) / (10 * kelvin))
correct_K = Q10_K ** ((temp - temp_exp) / (10 * kelvin))
correct_Na = Q10_Na ** ((temp - temp_exp) / (10 * kelvin))
correct_BK = Q_BK * (temp - temp_exp)
Ca_o = 2.5 * mM
Na_o = 137 * mM
Na_i = 10 * mM
K_o = 5.9 * mM
K_i = 164 * mM
Cm = 77 * pF
tau_x_a1_C6 = 2
nCa = -4
Ca_set_BK = 0.001 * mM
r_lig_C1 = 0.01 * mM
ACh = 1e-5 * mM
Vc = 3500 * um3
RToF = R * temp / F
EK = RToF * log(K_o / K_i)
ENa = RToF * log(Na_o / Na_i)

eqs = """
ECa = RToF * 0.5 * log(Ca_o / Ca_i) : volt
# CaL
I_CaL : amp # blocked
# CaT
I_CaT = G_CaT*d_CaT*f_CaT*(ECa-v) : amp
dd_CaT/dt = (d_CaT_inf-d_CaT)/(tau_d_CaT*2.62352425) : 1
d_CaT_inf = 1/(1+exp(-(v+V_half_d_CaT)/k_d_CaT)) : 1
df_CaT/dt = (f_CaT_inf-f_CaT)/(tau_f_CaT*2.62352425) : 1
f_CaT_inf = 1/(1+exp((v+V_half_f_CaT)/k_f_CaT)) : 1
tau_f_CaT = tau_f_CaT_C1*exp(tau_f_CaT_C2*v) : second
# Kr
I_Kr = G_Kr*x_r1*x_r2*(EK-v) : amp
dx_r1/dt = (x_r1_inf-x_r1)/(tau_x_r1*1.49855417) : 1
x_r1_inf = 1/(1+exp(-(v+V_half_x_r1)/k_x_r1)) : 1
dx_r2/dt = (x_r2_inf-x_r2)/(tau_x_r2*1.49855417) : 1
x_r2_inf = scale_x_r2/(1+exp((v+V_half_x_r2)/k_x_r2))+1-scale_x_r2 : 1
tau_x_r2 = -tau_x_r2_C1+tau_x_r2_C2*exp((v+tau_x_r2_C3)/tau_x_r2_C4) : second
# Ka
I_Ka = G_Ka*x_a1*x_a2*(EK-v) : amp
dx_a1/dt = (x_a1_inf-x_a1)/(tau_x_a1*1.49855417) : 1
x_a1_inf = 1/(1+exp(-(v+V_half_x_a1)/k_x_a1)) : 1
tau_x_a1 = tau_x_a1_C1+tau_x_a1_C2*exp(tau_x_a1_C3*((v+tau_x_a1_C4)/tau_x_a1_C5)**tau_x_a1_C6) : second
dx_a2/dt = (x_a2_inf-x_a2)/(tau_x_a2*1.49855417) : 1
x_a2_inf = scale_x_a2/(1+exp(-(v+V_half_x_a2)/k_x_a2))+1-scale_x_a2 : 1
# BK
I_BK = (G_BK+14.3*nS)*P0_BK*(EK-v) : amp
P0_BK = 1/(1+exp(v/K_BK-h_BK*log(Ca_i/Ca_set_BK))) : 1
# Kb
I_Kb = G_Kb*(EK-v) : amp
# Na
I_Na = G_Na*d_Na*f_Na*(ENa-v) : amp
dd_Na/dt = (d_Na_inf-d_Na)/(tau_d_Na*3.20564857) : 1
d_Na_inf = 1/(1+exp(-(v+V_half_d_Na)/k_d_Na)) : 1
tau_d_Na = tau_d_Na_C1*v+tau_d_Na_C2 : second
df_Na/dt = (f_Na_inf-f_Na)/(tau_f_Na*3.20564857) : 1
f_Na_inf = 1/(1+exp((v+V_half_f_Na)/k_f_Na)) : 1
tau_f_Na = tau_f_Na_C1*v+tau_f_Na_C2 : second
# NCX
nom_NCX = 2.5*exp((gamma_NCX-1)*v/RToF)*Na_o**3*Ca_i - exp(gamma_NCX*v/RToF)*Na_i**3*Ca_o: mM**4
denom_NCX = (K_mNa_i_NCX**3+Na_o**3)*(K_mCa_NCX+Ca_o)*(1+k_sat_NCX*exp((gamma_NCX-1)*v/RToF)) : mM**4
I_NCX = nom_NCX / denom_NCX * P_NCX : amp
# Ca2+ dynamics
dCa_i/dt = (I_CaL+I_CaT-2*I_NCX)/(2*F*Vc)/(1+CRT_tot*K_CRT_D/(Ca_i+K_CRT_D)**2+
            4*CaM_tot*K_CaM_D*Ca_i**3/(Ca_i**4+K_CaM_D)**2) : mM
# gap junction
I_GJ = (v_ICC-v)*G_GJ_SMC_ICC : amp
# membrane
dv/dt = (I_CaL+I_CaT+I_Kr+I_Ka+I_BK+I_Kb+I_Na+I_GJ+I_NCX)/Cm : volt

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
# Ca2+ dynamics
# Ca_Ext_C1: mM / second (constant)
# Ca_Ext_C2: 1 (constant)
CRT_tot : mM (constant)
K_CRT_D : mM (constant)
CaM_tot : mM (constant)
K_CaM_D : mM**4 (constant)
"""
fitter = TraceFitter(
    model=eqs,
    input_var='v_ICC',
    output_var='v',
    input=v_ICC,
    output=v_SMC_exper,
    dt=0.1 * ms,
    n_samples=100,
    method='euler',
    param_init=init
)

res, error = fitter.fit(
    n_rounds=400,
    optimizer=NevergradOptimizer(),
    metric=MSENaNMetric(1000, t_weights=t_wt),
    callback=callback,
    # CaT
    V_half_d_CaT=[10 * mV, 25 * mV], k_d_CaT=[5 * mV, 17 * mV],
    V_half_f_CaT=[8 * mV, 22 * mV], k_f_CaT=[4 * mV, 8.3 * mV],
    tau_d_CaT=[1 * ms, 5 * ms], tau_f_CaT_C1=[1 * ms, 15 * ms], tau_f_CaT_C2=[1e-3 / mV, 3e-2 / mV],
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
    k_sat_NCX=[0.1, 0.7], P_NCX=[3000 * pA, 8000 * pA],
    # Ca2+ dynamics
    CRT_tot=[0.005 * mM, 0.04 * mM], K_CRT_D=[1e-3 * mM, 1e-2 * mM], CaM_tot=[0.02 * mM, 0.15 * mM],
    K_CaM_D=[1e-4 * mM ** -5, 8e-4 * mM ** 4]
)
trace1 = fitter.generate_traces()
ts = arange(v_SMC_exper.shape[1]) * 0.1
plot(ts, v_SMC_exper[0] / mV, label='exp')
plot(ts, trace1[0] / mV, label='pred1')
show()
history = pd.DataFrame(params_all)
history['error'] = error_all
for name, unit in prop_units.items():
    history[name] = history[name].apply(lambda x: x / unit)
history = history[['error', *prop_units.keys()]]
with open('SMC-opt-history.pkl', 'wb') as f1:
    pickle.dump(history, f1)
with open('SMC-params0.pkl', 'wb') as f:
    pickle.dump({'params': res}, f)
