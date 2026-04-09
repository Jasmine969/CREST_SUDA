from brian2.only import *
from brian2 import numpy_ as np
from brian2 import codegen
from utils.result_path import RES_PATH
import pickle

codegen.cpp_prefs._compiler_supports_c99 = True
BrianLogger.suppress_hierarchy('brian2.codegen.generators.base')
"""
delete ICC
"""


def PhysiolModel(
        go_on,
        force_factor=1e-4,
        epsilon_half=0.07,
        JPalpha=90,
        V_half_EJP=-45, V_half_IJP=-60,
        w_boundary1=1,
        w_boundary2=1,
        neuron_pop=146,
        N_callback_net=10,
        dt_net=1e-4, dt_couple=1e-3,
        n_syn_inh=4, n_syn_exc=2
):
    defaultclock.dt = dt_net * second
    # Parameters ==================================================================
    muscle_pop = neuron_pop * 2
    # global params
    p_global = dict(
        R=8.3144626,  # joule / (mole * kelvin),
        F=96485.332,  # coulomb / mole,
        temp=310.15  # kelvin
    )
    p_global['RToF'] = p_global['R'] * p_global['temp'] / p_global['F']
    # AH-type neuron
    p_AH = dict(
        Cm=200 * pF,
        G_leak=0.06 * uS, G_Na=24 * uS, G_K=4 * uS, G_KA=9.54 * uS,
        EL=-17.0 * mV, EK=-72 * mV, ENa=55.0 * mV, E_KA=-75 * mV,
        # AH
        E_AH=-89 * mV,
        T_on=0.07 * second, T_off=0.5 * second,
        # SAC
        E_SAC=40 * mV, G_SAC=0.05 * uS, epsilon_half=epsilon_half
    )
    # S-type neuron
    p_S = dict(
        Cm=200 * pF,
        ENa=55 * mV, EK=-85 * mV, EL=-55 * mV,
        G_Nav13=33.72 * uS, G_Kdr=77.64 * uS, G_Kv72=16.574 * uS, G_leak=1.2668 * nS,
        # synapse
        w_std=2.5,
        E_EPSP=0 * mV, gmaxEPSP=10 * uS,
        wEPSP=0.01305, tauEPSP1=5 * ms, tauEPSP2=1 * ms
    )
    p_S.update(p_global)
    p_S.update(dict(
        q_Nav13=2 ** ((p_S['temp'] - (24 + 273.15)) / 10),
        q_Kv72=5 ** ((p_S['temp'] - (22 + 273.15)) / 10)
    ))
    # SMC
    p_SMC = dict(
        temp_exp=297,
        Q10_Ca=2.1, Q10_K=1.365, Q10_Na=2.45, Q_BK=1.1 * nS,
        Ca_o=2.5 * mM, Na_o=137 * mM, Na_i=10 * mM, K_o=5.9 * mM, K_i=164 * mM,
        Cm=77 * pF, Vc=3500 * um3,
        # CaL
        V_half_d_CaL=17 * mV, k_d_CaL=2.69193236 * mV, tau_d_CaL=0.55267451 * ms,
        V_half_f_CaL=60.42 * mV, k_f_CaL=18.90 * mV, tau_f_CaL=31.24 * ms,
        h_Ca_CaL=0.977789 * uM, s_Ca_CaL=12.134 * nM, tau_fCa_CaL=1.035 * ms,
        G_CaL=0.0744025 * uS,
        # CaT
        V_half_d_CaT=44.50885215 * mV, k_d_CaT=5.903 * mV,
        V_half_f_CaT=74.7018577 * mV, k_f_CaT=8.21486707 * mV,
        tau_d_CaT=2.77953049 * ms, tau_f_CaT_C1=25.80621858 * ms, tau_f_CaT_C2=0.04884692 / mV,
        G_CaT=3.362232386e-5 * uS,
        # Kr
        V_half_x_r1=32.17370263 * mV, k_x_r1=1.68665541 * mV,
        V_half_x_r2=42.65304 * mV, k_x_r2=2.17009 * mV, scale_x_r2=0.50786823,
        tau_x_r1=113.5243179 * ms, tau_x_r2_C1=428.93328 * ms, tau_x_r2_C2=1922.56769 * ms,
        tau_x_r2_C3=49.43162179 * mV, tau_x_r2_C4=63.31677651 * mV,
        G_Kr=0.028 * uS,
        # Ka
        V_half_x_a1=14.308238 * mV, k_x_a1=18.98043929 * mV,
        V_half_x_a2=71.238839 * mV, k_x_a2=4.11510969 * mV, scale_x_a2=0.874379,
        tau_x_a1_C1=34.5655251 * ms, tau_x_a1_C2=397.65227 * ms, tau_x_a1_C3=-0.9620667556,
        tau_x_a1_C4=23.57293609 * mV, tau_x_a1_C5=33.18094175 * mV, tau_x_a1_C6=2,
        tau_x_a2=88.81762256 * ms,
        G_Ka=8.0602379e-4 * uS,
        # BK
        h_BK=7.48663546942, K_BK=-22.41 * mV, Ca_set_BK=0.001 * mM,
        G_BK=0.02234840821 * uS,
        # Kb
        G_Kb=1.668e-5 * uS,
        # Na
        V_half_d_Na=45.91644605 * mV, k_d_Na=3.06621979 * mV,
        V_half_f_Na=80.14653663 * mV, k_f_Na=3.95498271 * mV,
        tau_d_Na_C1=-0.00535491 / mV * ms, tau_d_Na_C2=220.12535501 * us,
        tau_f_Na_C1=-0.58614092 / mV * ms, tau_f_Na_C2=6.80416264 * ms,
        G_Na=4.89241048e-3 * uS,
        # NCX
        gamma_NCX=0.36245388, K_mNa_i_NCX=95.520285 * mM, K_mCa_NCX=0.456232 * mM,
        k_sat_NCX=0.695352, P_NCX=4.02635421 * nA,
        # MLCK activation
        kf1=12 / (uM ** 2 * second), kr1=12 / second,
        kf2=480 / (uM ** 2 * second), kr2=1200 / second,
        kf3=5 / (uM * second), kr3=135 / second,
        kf4=840 / (uM * second), kr4=45.4 / second,
        kf5=28 / (uM * second), kr5=0.0308 / second,
        kf6=120 / (uM ** 2 * second), kr6=4 / second,
        kf7=7.5 / (uM ** 2 * second), kr7=3.75 / second,
        kf8=5 / (uM * second), kr8=25 / second,
        kf9=7.6 / (uM ** 2 * second), kr9=22.8 / (uM * second),
        # cross-bridge formation
        k_cat_MLCK=27 / second, Km_MLCK=10 * uM,
        k3cb=15 / second, k4cb=5 / second,
        k_cat_MLCP=16 / second, Km_MLCP=15 * uM, k7cb=10 / second,
        MLCP=7.5 * uM, M_total=24 * uM,
        # Synapse
        E_EJP=0 * mV, gmaxEJP=0.3 * uS,
        wEJP=0.01305, tauEJP1=30 * ms, tauEJP2=6 * ms,
        E_IJP=-80 * mV, gmaxIJP=0.9 * uS,
        wIJP=0.01305, tauIJP1=30 * ms, tauIJP2=6 * ms,
        w_cGMP=3, tau_cGMP=100 * ms,
        V_halfEJP=V_half_EJP, V_halfIJP=V_half_IJP,
        JPalpha=JPalpha
    )
    p_SMC.update(p_global)
    p_SMC.update(dict(
        EK=p_SMC['RToF'] * log(p_SMC['K_o'] / p_SMC['K_i']) * volt,
        ENa=p_SMC['RToF'] * log(p_SMC['Na_o'] / p_SMC['Na_i']) * volt,
        T_correct_Ca=p_SMC['Q10_Ca'] ** ((p_SMC['temp'] - p_SMC['temp_exp']) / 10),
        T_correct_K=p_SMC['Q10_K'] ** ((p_SMC['temp'] - p_SMC['temp_exp']) / 10),
        T_correct_Na=p_SMC['Q10_Na'] ** ((p_SMC['temp'] - p_SMC['temp_exp']) / 10),
        T_correct_BK=p_SMC['Q_BK'] * (p_SMC['temp'] - p_SMC['temp_exp'])
    ))
    p_SMC['N_callback_net'] = N_callback_net
    p_SMC['force_factor'] = force_factor

    # Cell definitions ======================================================================
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
    I_SAC = G_SAC/(1+exp(-(DSTND-epsilon_half)/0.003))*(E_SAC-v)*int(DSTND>0) : amp
    DSTND = distension(t,i) : 1
    dv/dt = (I_Na+I_K+I_KA+I_AH+I_leak+I_SAC)/Cm : volt
    '''
    IPAN = NeuronGroup(
        neuron_pop, eqs_AH, method='exponential_euler',
        threshold='v>10*mV',
        reset='zAH+=50*nS/second',
        refractory=1 * ms,
        namespace=p_AH, name='IPAN')
    if not go_on:
        IPAN.v = -65 * mV
        IPAN.m_Na = 'alpha_m_Na/(alpha_m_Na+beta_m_Na)'
        IPAN.h_Na = 'alpha_h_Na/(alpha_h_Na+beta_h_Na)'
        IPAN.n_K = 'alpha_n_K/(alpha_n_K+beta_n_K)'
        IPAN.a_KA = 'a_inf_KA'
        IPAN.b_KA = 'b_inf_KA'
        IPAN.gAH = 0.08 * uS

    # S-type neuron
    eqs_S = '''
    # Nav13
    I_Nav13 = G_Nav13*m_Nav13**3*h_Nav13*(ENa-v) : amp
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
    I_Kdr = G_Kdr*n_Kdr**4*(EK-v) : amp
    dn_Kdr/dt = (n_inf_Kdr-n_Kdr)/tau_n_Kdr : 1
    tau_n_Kdr = (0.25+4.35*exp(-sign(v+10*mV)*(v/mV+70)/15))*ms : second
    n_inf_Kdr = 1/(1+exp(-(v/mV)/25)) : 1

    # Kv72
    I_Kv72 = G_Kv72 * m1_Kv72**3 * m2_Kv72 * (EK-v) : amp
    # z1*gamma*frt == 16.2463, z2*gamma*frt == 51.64
    dm1_Kv72/dt = (m_inf_Kv72-m1_Kv72)/tau_m1_Kv72 : 1
    dm2_Kv72/dt = (m_inf_Kv72-m2_Kv72)/tau_m2_Kv72 : 1
    tau_m1_Kv72 = (176.1*0.5/cosh((v/mV+20)*16.2463/temp)+20.7)/q_Kv72*ms : second
    tau_m2_Kv72 = (1473*0.5/cosh((v/mV+20)*51.64/temp)+149)/q_Kv72*ms : second
    m_inf_Kv72 = 1/(1+exp(-(v/mV+20)/18.4)) : 1

    # leak channel
    I_leak = G_leak*(EL-v) : amp

    # synapse
    I_EPSP = gEPSP*gmaxEPSP*(E_EPSP-v) : amp
    dgEPSP/dt = ((tauEPSP2/tauEPSP1)**(tauEPSP1/(tauEPSP2-tauEPSP1))*zEPSP-gEPSP)/tauEPSP1 : 1
    dzEPSP/dt = -zEPSP/tauEPSP2 : 1
    dv/dt = (I_Nav13+I_Kdr+I_Kv72+I_leak+I_EPSP)/Cm : volt

    du_NO/dt = -u_NO/(10*ms)       : 1
    dx_NO/dt = (1 - x_NO)/(100*ms) : 1
    '''
    SN = NeuronGroup(
        neuron_pop * 4, eqs_S,
        threshold='v>10*mV', refractory=4 * ms,
        method='exponential_euler', namespace=p_S, name='SN'
    )
    if not go_on:
        SN.v = -70 * mV
        SN.m_Nav13 = 'm_inf_Nav13'
        SN.h_Nav13 = 'h_inf_Nav13'
        SN.n_Kdr = 'n_inf_Kdr'
        SN.m1_Kv72 = 'm_inf_Kv72'
        SN.m2_Kv72 = 'm_inf_Kv72'
        SN.x_NO = 1
    AIN = SN[:neuron_pop]
    DIN = SN[neuron_pop:neuron_pop * 2]
    ECMN = SN[neuron_pop * 2:neuron_pop * 3]
    ICMN = SN[neuron_pop * 3:neuron_pop * 4]

    eqs_SMC = '''
    # electrophysiology ===============================================================================
    ECa = RToF*0.5*log(Ca_o/Ca_i)*volt : volt
    # CaL
    I_CaL = G_CaL*d_CaL*f_CaL*fCa_CaL*(ECa-v) : amp
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
    nom_NCX = 2.5*exp((gamma_NCX-1)*v/(RToF*volt))*Na_o**3*Ca_i - exp(gamma_NCX*v/(RToF*volt))*Na_i**3*Ca_o: mM**4
    denom_NCX = (K_mNa_i_NCX**3+Na_o**3)*(K_mCa_NCX+Ca_o)*(1+k_sat_NCX*exp((gamma_NCX-1)*v/(RToF*volt))) : mM**4
    I_NCX = nom_NCX / denom_NCX * P_NCX : amp
    # Ca2+ dynamics
    dCa_i/dt = (I_CaL+I_CaT-2*I_NCX)/(2*F*coulomb/mole*Vc) : mM
    # gap junction
    I_GJ_SMC_SMC : amp
    # membrane
    dv/dt = (I_CaL+I_CaT+I_Kr+I_Ka+I_BK+I_Kb+I_Na+I_NCX+I_GJ_SMC_SMC+I_EJP+I_IJP)/Cm : volt
    # synapse
    w_boundary_IJP : 1
    w_boundary_EJP : 1
    I_EJP = gEJP*gmaxEJP*(E_EJP-v) : amp
    dgEJP/dt = ((tauEJP2/tauEJP1)**(tauEJP1/(tauEJP2-tauEJP1))*zEJP-gEJP)/tauEJP1 : 1
    dzEJP/dt = -zEJP/tauEJP2 : 1
    I_IJP = gIJP*gmaxIJP*(E_IJP-v) : amp
    dgIJP/dt = ((tauIJP2/tauIJP1)**(tauIJP1/(tauIJP2-tauIJP1))*zIJP-gIJP)/tauIJP1 : 1
    dzIJP/dt = -zIJP/tauIJP2 : 1
    EJPalpha = (JPalpha / (1 + exp(-(v / mV - V_halfEJP))) + 100 - JPalpha) / 100 : 1
    IJPalpha = (JPalpha / (1 + exp(-(v / mV - V_halfIJP))) + 100 - JPalpha) / 100 : 1
    allow_spike: 1
    
    dcGMP/dt = (1-cGMP)/tau_cGMP : 1

    # MLCK activation =============================
    Ca_eff = Ca_i / cGMP : mM
    # items
    f1 = kf1 * CaM          * Ca_eff**2 : mM/second
    r1 = kr1 * Ca2CaM                   : mM/second
    f2 = kf2 * Ca2CaM       * Ca_eff**2 : mM/second
    r2 = kr2 * Ca4CaM                   : mM/second
    f3 = kf3 * CaM          * MLCK      : mM/second
    r3 = kr3 * CaM_MLCK                 : mM/second
    f4 = kf4 * Ca2CaM       * MLCK      : mM/second
    r4 = kr4 * Ca2CaM_MLCK              : mM/second
    f5 = kf5 * Ca4CaM       * MLCK      : mM/second
    r5 = kr5 * Ca4CaM_MLCK              : mM/second
    f6 = kf6 * CaM_MLCK     * Ca_eff**2 : mM/second
    r6 = kr6 * Ca2CaM_MLCK              : mM/second
    f7 = kf7 * Ca2CaM_MLCK  * Ca_eff**2 : mM/second
    r7 = kr7 * Ca4CaM_MLCK              : mM/second
    f8 = kf8 * CaM * BP                 : mM/second
    r8 = kr8 * CaM_BP                   : mM/second
    f9 = kf9 * CaM_BP       * Ca_eff**2 : mM/second
    r9 = kr9 * Ca2CaM       * BP        : mM/second
    # reactions
    dCaM/dt = -f1+r1-f3+r3-f8+r8        : mM
    # dCa_i/dt =  : mM
    dCa2CaM/dt = f1-r1-f2+r2-f4+r4+f9-r9: mM
    dCa4CaM/dt = f2-r2-f5+r5            : mM
    dMLCK/dt = -f3+r3-f4+r4-f5+r5       : mM
    dCaM_MLCK/dt = f3-r3-f6+r6          : mM
    dCa2CaM_MLCK/dt = f4-r4+f6-r6-f7+r7 : mM
    dCa4CaM_MLCK/dt = f5-r5+f7-r7       : mM
    dBP/dt = -f8+r8+f9-r9               : mM
    dCaM_BP/dt = f8-r8-f9+r9            : mM
    # cross-bridge formation ==============================================
    # items
    K1 = k_cat_MLCK * Ca4CaM_MLCK   * M     / (Km_MLCK + M)     : mM/second
    K2 = k_cat_MLCP * MLCP          * Mp    / (Km_MLCP + Mp)    : mM/second
    K3 = k3cb                       * Mp                        : mM/second
    K4 = k4cb                       * AMp                       : mM/second
    K5 = k_cat_MLCP * MLCP          * AMp   / (Km_MLCP + AMp)   : mM/second
    K6 = k_cat_MLCK * Ca4CaM_MLCK   * AM    / (Km_MLCK + AM)    : mM/second
    K7 = k7cb                       * AM                        : mM/second
    # reactions
    dM/dt = -K1+K2+K7       : mM
    dMp/dt = K1-K2-K3+K4    : mM
    dAMp/dt = K3-K4-K5+K6   : mM
    dAM/dt = K5-K6-K7       : mM
    T = (AM+AMp)/M_total*force_factor    : 1
    T_avg : 1
    '''
    SMC = NeuronGroup(
        muscle_pop, eqs_SMC, method='euler',
        namespace=p_SMC, name='SMC'
    )
    if not go_on:
        SMC.w_boundary_EJP = 1
        SMC.w_boundary_EJP[:2] = w_boundary1
        SMC.w_boundary_EJP[2:4] = w_boundary2
        SMC.w_boundary_IJP = 1
        SMC.w_boundary_IJP[-2:] = w_boundary1
        SMC.w_boundary_IJP[-4:-2] = w_boundary2
        SMC.v = -74 * mV
        SMC.Ca_i = 8e-5 * mM
        SMC.f_CaL = 'f_CaL_inf'
        SMC.fCa_CaL = 'fCa_CaL_inf'
        SMC.d_CaT = 'd_CaT_inf'
        SMC.f_CaT = 'f_CaT_inf'
        SMC.x_r2 = 'x_r2_inf'
        SMC.x_a1 = 'x_a1_inf'
        SMC.x_a2 = 'x_a2_inf'
        SMC.d_Na = 'd_Na_inf'
        SMC.f_Na = 'f_Na_inf'
        SMC.CaM = 0.9285 * uM
        SMC.MLCK = 9.6506 * uM
        SMC.Ca2CaM = 0.0015 * uM
        SMC.Ca4CaM = 0 * uM
        SMC.CaM_MLCK = 0.3332 * uM
        SMC.Ca2CaM_MLCK = 0.2713 * uM
        SMC.Ca4CaM_MLCK = 0.0130 * uM
        SMC.CaM_BP = 2.8207 * uM
        SMC.BP = 15.1793 * uM
        SMC.M = 23.9558 * uM
        SMC.Mp = 0.0144 * uM
        SMC.AMp = 0.0166 * uM
        SMC.AM = 0.0132 * uM
        SMC.T_avg = 'T'
        SMC.cGMP = 1
        SMC.allow_spike = 1
    SMC.run_regularly('''
        T_avg += T / N_callback_net
    ''', when='after_groups', dt=dt_net * second)
    SMC.run_regularly('T_avg = 0', dt=dt_couple * second, when='after_start')

    # Synapses ================================================================
    IPAN_AIN = Synapses(
        IPAN, AIN, delay=1 * ms, on_pre='zEPSP_post += wEPSP',
        namespace=p_S, name='IPAN_AIN')
    IPAN_AIN.connect(j='i')

    IPAN_DIN = Synapses(
        IPAN, DIN, delay=1 * ms, on_pre='zEPSP_post += wEPSP',
        namespace=p_S, name='IPAN_DIN')
    IPAN_DIN.connect(j='i')

    AIN_ECMN = Synapses(
        AIN, ECMN, 'w = exp(-(i - j) ** 2 / (2 * w_std ** 2)) : 1',
        delay=20 * ms, on_pre='zEPSP_post += wEPSP * w',
        namespace=p_S, name='AIN_ECMN'
    )
    AIN_ECMN.connect(j=f'i-k for k in range(1,{n_syn_exc+1})', skip_if_invalid=True)

    DIN_ICMN = Synapses(
        DIN, ICMN, 'w = exp(-(i - j) ** 2 / (2 * w_std ** 2)) : 1',
        delay=20 * ms, on_pre='zEPSP_post += wEPSP * w',
        namespace=p_S, name='DIN_ICMN'
    )
    DIN_ICMN.connect(j=f'i+k for k in range(1,{n_syn_inh+1})', skip_if_invalid=True)

    ECMN_SMC = Synapses(
        ECMN, SMC, delay=1 * ms, on_pre='zEJP_post += wEJP*EJPalpha*w_boundary_EJP*allow_spike_post',
        namespace=p_SMC, name='ECMN_SMC'
    )
    ECMN_SMC.connect(i='j//2')

    ICMN_SMC = Synapses(
        ICMN, SMC, delay=1 * ms,
        on_pre='''
        u_NO_pre += 0.2 * (1 - u_NO_pre)
        cGMP_post += u_NO_pre*x_NO_pre*w_cGMP*w_boundary_IJP
        x_NO_pre -= u_NO_pre*x_NO_pre
        zIJP_post += wIJP*IJPalpha*w_boundary_IJP
        ''',
        namespace=p_SMC, name='ICMN_SMC'
    )
    ICMN_SMC.connect(i='j//2')

    GJ_SMC_SMC = Synapses(SMC, SMC,
                          '''I_GJ_SMC_SMC_pre = 10 * nS * (v_post - v_pre) : amp (summed)''',
                          name='GJ_SMC_SMC')
    GJ_SMC_SMC.connect(condition='abs(i - j) == 1')

    # create network
    network = Network(
        IPAN, SN,
        SMC,
        GJ_SMC_SMC,
        IPAN_AIN, IPAN_DIN,
        AIN_ECMN, DIN_ICMN,
        ECMN_SMC,
        ICMN_SMC
    )

    monitor_dt = defaultclock.dt * 5  # record every 5 dt

    mSMC = StateMonitor(SMC, (
        'v', 'gEJP', 'zEJP', 'Ca_i',
        'I_CaL', 'I_CaT', 'I_Kr', 'I_Ka', 'I_BK', 'I_Kb', 'I_Na', 'I_NCX',
        'I_GJ_SMC_SMC', 'I_EJP', 'I_IJP', 'cGMP',
        'T'
    ), record=True, name='mSMC', dt=monitor_dt, when='start')
    mTavg = StateMonitor(SMC, 'T_avg', record=True, name='mTavg', dt=dt_couple * second, when='start')
    mSN = StateMonitor(
        SN, (
            'v',
            'I_Nav13', 'I_Kdr', 'I_Kv72', 'I_leak', 'I_EPSP'
        ),
        record=True, name='mSN', dt=monitor_dt, when='start')
    mIPAN = StateMonitor(IPAN, 'v', record=True, name='mIPAN', dt=monitor_dt, when='start')
    mStrain = StateMonitor(IPAN, 'DSTND', record=True, name='mStrain', dt=dt_couple * second, when='start')
    sSN = SpikeMonitor(SN, record=True, name='sSN')
    sIPAN = SpikeMonitor(IPAN, record=True, name='sIPAN')

    network.add(
        mSMC,
        mSN,
        mIPAN,
        sSN,
        sIPAN,
        mTavg, mStrain
    )
    return network


if __name__ == '__main__':
    case_name = 'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
    N_callback_lmp = 50
    go_on_from_step = 0
    read_step = 1250000
    dt_lmp = 2e-5 * second
    with open(f'{RES_PATH}/{case_name}/net_params.pkl', 'rb') as f:
        net_params = pickle.load(f)
    for key, value in net_params.items():
        print(f'{key}: {value}')
    ring_sense_start = 8
    # net_params['w_boundary1'] = 1
    # net_params['w_boundary2'] = 1
    # net_params['EJP_alpha'] = 90
    # net_params['IJP_alpha'] = 90
    # del net_params['EJP_alpha']
    # del net_params['IJP_alpha']
    # net_params['V_half_EJP'] = -45
    # net_params['V_half_IJP'] = -60
    net = PhysiolModel(False, **net_params)
    st = np.load(f'{RES_PATH}/{case_name}/interface/strain_tension_{read_step}.npz')
    strain = st['strain'][:, ring_sense_start:-ring_sense_start].reshape(25000, 92, 2).mean(axis=-1)
    distension = TimedArray(strain[int(go_on_from_step / N_callback_lmp):], dt=1 * ms)
    print(distension.values.shape)
    # distension = TimedArray(np.tile(st['strain'][0], (4000, 1)), dt=1 * ms)
    if go_on_from_step:
        with open(f'{RES_PATH}/{case_name}/state/net_{go_on_from_step}.pkl', 'rb') as pf_state:
            states = pickle.load(pf_state)
        for objects in states.values():
            for k, val in objects.items():
                if k.startswith("lastspike"):
                    val -= 1 * ksecond
        net.set_states(states)
    net.run(dt_lmp * (read_step - go_on_from_step), report='text')

    net.store(filename=f'{RES_PATH}/{case_name}/store/net_{go_on_from_step}to{read_step}.store')
    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(2, 1, sharex=True)
    # id_smc = 2
    # ax[0].plot(net['mSMC'].t / ms, net['mSMC'].v[id_smc] / mV)
    # ax[1].plot(net['mSMC'].t / ms, net['mSMC'].T[id_smc])
    # plt.figure(2)
    # im = plt.imshow(net['mSMC'].T, aspect='auto')
    # plt.colorbar(im)
    # net.restore(filename=f'{RES_PATH}/{case_name}/store/net_{go_on_from_step}to{read_step}.store')
    # ax[0].plot(net['mSMC'].t / ms, net['mSMC'].v[id_smc] / mV)
    # ax[1].plot(net['mSMC'].t / ms, net['mSMC'].T[id_smc])
    # plt.show()
