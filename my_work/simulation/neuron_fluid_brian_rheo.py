import os
import pickle
import sys
from pathlib import Path
from socket import gethostname
from time import time, asctime
from warnings import warn

import yaml
from brian2 import numpy_ as np
from brian2.units import second, mV
from lammps import (lammps, LMP_VAR_EQUAL, LMP_ERROR_ALL,
                    LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY, c_int)
from mpi4py import MPI

root_path = Path(os.getenv('MY_WORK')).parent
sys.path.append(str(root_path))
from my_work.network_models.Physiol_Model_noICC import PhysiolModel
#from my_work.network_models.Physiol_Model_noICC_old import PhysiolModel
from utils.utils_misc import format_time
from utils.safefy import check_and_create_lock

comm = MPI.COMM_WORLD
me = comm.Get_rank()

case_name = 'rheo_bond2_angle-F100-krebs-noICC-28w-ringstrain'
n_yz = 63
n_inlet = 3870
n_wall = 12600
dt_net = 1e-4
r_si = 0.002
# report error if non-digit-letter-_ is in case_name
# assert not search('\\w', case_name)
case_path = f'../results/{case_name}'

# read global configurations from yaml file
p_global = None
if me == 0:
    with open(f'../results/{case_name}.yml', 'r') as f:
        p_global = yaml.safe_load(f)
p_global = comm.bcast(p_global, root=0)
assert case_name == p_global['case_name']
dt_lmp = p_global['dt_lmp']
# save restart and state files every this time_step
save_interval = p_global['save_interval']
dmp_interval = p_global['dmp_interval']
# display params
mylog_interval = p_global['mylog_interval']
# couple params
force_factor = p_global['force_factor']
N_callback_lmp = p_global['N_callback_lmp']
wave_interval_min = p_global['wave_interval_min']
# assert dmp_interval % N_callback_lmp == 0
dt_couple = dt_lmp * N_callback_lmp
N_callback_net = int(dt_couple / dt_net)
assert N_callback_net * dt_net == dt_couple
n_ring = int(n_wall / n_yz)
assert n_wall == n_yz * n_ring
n_sense_each = p_global['n_sense_each']
ring_sense_start = p_global['ring_sense_start']  # 8
ring_sense_end = n_ring - 1 - ring_sense_start  # 191
n_sense = int((ring_sense_end + 1 - ring_sense_start) / n_sense_each)  # 92
n_muscle_each = int(n_sense_each / 2)
assert n_muscle_each * 2 == n_sense_each
id_sense_start = int(n_inlet + ring_sense_start * n_yz + 1)
id_sense_end = int(n_inlet + (ring_sense_end + 1) * n_yz)
id_wall = np.arange(n_inlet, n_inlet + n_wall) + 1
assert id_wall[n_yz * ring_sense_start] == id_sense_start
assert id_wall[n_yz * (ring_sense_end + 1) - 1] == id_sense_end
id_wall = (n_wall * c_int)(*id_wall)
peri0 = 2 * np.pi * r_si
# network params
net_params = None
if me == 0:
    net_params = {
        'neuron_pop': n_sense,
        'force_factor': force_factor,
        'epsilon_half': p_global['DSTNDhalf'],
        'JPalpha': p_global['JPalpha'],
        'V_half_EJP': p_global['V_half_EJP'],
        'V_half_IJP': p_global['V_half_IJP'],
        'w_boundary1': p_global['w_boundary1'],
        'w_boundary2': p_global['w_boundary2'],
        'N_callback_net': N_callback_net,
        'dt_couple': dt_couple,
        'dt_net': dt_net,
        'n_syn_inh': p_global['n_syn_inh'],
        'n_syn_exc': p_global['n_syn_exc']
    }

# go_on_from_step==0 if ab initio
go_on_from_step = p_global['go_on_from_step']
n_step = p_global['n_step']
end_step = go_on_from_step + n_step
if me == 0 and n_step // save_interval > 200:
    warn('There will be over 200 result files.'
         ' Change save_interval or continue? (y/n)\n')
    if_continue = input()
    if if_continue.lower() in ['n', 'no', 'false']:
        raise Exception
if me == 0:
    # restart: lammps, state: brian2, interface: strain + tension,
    # restart_params: state scalars like t_since_activated
    for directory in ['restart', 'state', 'interface', 'restart_params']:
        os.makedirs(f'{case_path}/{directory}', exist_ok=True)
    check_and_create_lock(case_path, case_name)
comm.Barrier()
if me == 0:
    print('See my.log')
    with open(f'{case_path}/my{go_on_from_step}_to{end_step}.log', 'w') as f:
        print(f'case: {case_name}\nRunning on {gethostname()}'
              f'\n{asctime()}\n\n', file=f)

# multiphysics
lmp = lammps(cmdargs=[
    '-screen', 'none',
    '-log', f'{case_path}/log_{go_on_from_step}to{end_step}.lammps']
)
# lmp = lammps()
lmp.commands_string(f"""
variable case_path string {case_path}
variable case_name string {case_name}
variable dmp_interval equal {dmp_interval}
variable N_restart equal {save_interval}
variable go_on_from_step equal {go_on_from_step}
variable end_step equal {end_step}
variable n_sense_each equal {n_sense_each}
variable Ncallback equal {N_callback_lmp}
variable soft equal {p_global['soft']}
variable k1_self equal {p_global['k1_self']}
variable k2_self equal {p_global['k2_self']}
variable kB1  equal {p_global['kB_circ']}
variable kB2  equal {p_global['kB_long']}
variable kA equal {p_global['kA']}
variable Dt equal {dt_lmp}
""")

lmp.file('head-rheo.in')
balance = p_global['balance']
assert balance in ['rcb', 'shift', False]
if comm.size == 1:
    balance = False
if not balance:
    lmp.command('processors * 1 1')
# neurons
net = None
if me == 0:
    net = PhysiolModel(go_on_from_step, **net_params)
if go_on_from_step:
    lmp.command(f'read_restart {case_path}/restart/{go_on_from_step}.restart')
    if me == 0:
        """
        I use get_states/set_states instead of store/restore here, because the former
        is more flexible; also, the file output by `store` in the MPI environment on the server 
        cannot be read successfully on my Windows 10 PC. See the following link for details:
        https://brian.discourse.group/t/synapses-cannot-be-simulated-after-using-set-states/1475/3
        """
        with open(f'{case_path}/state/net_{go_on_from_step}.pkl', 'rb') as pf_state:
            states = pickle.load(pf_state)
            # must reset lastspike so that IPAN will not be refractory, see the above link for details
            for objects in states.values():
                for k, val in objects.items():
                    if k.startswith("lastspike"):
                        val -= 1000 * second
            net.set_states(states)
    st = np.load(f'{case_path}/interface/strain_tension_{go_on_from_step}.npz')
    strains = list(st['strain'])
    strain0 = strains[-1][ring_sense_start:ring_sense_start + n_sense_each].mean()
    allow_spikes = list(st['allow_spikes'])
    tensions = list(st['tension'])
    with open(f'{case_path}/restart_params/state_scalars_{go_on_from_step}.yml', 'r') as ymf:
        state_scalars = yaml.safe_load(ymf)
        t_since_activated = state_scalars['t_since_activated']
else:
    lmp.commands_string(f"""
    read_restart ../create_geometry/data/{p_global['restart_file']}
    reset_timestep 0
    """)
    strains = []
    strain0 = 0
    tensions = []
    allow_spikes = []
    t_since_activated = 100  # second
    if me == 0:
        with open(f'{case_path}/net_params.pkl', 'wb') as pf_net:
            pickle.dump(net_params, pf_net)

lmp.file('bio_intestine_rheo.in')
if 'power' in p_global['restart_file']:
    lmp.command('fix visc all rheo/viscosity * power_openfoam 5e-4 5e-2 0.105 0.366')
else:
    lmp.commands_string(f"fix visc all rheo/viscosity * constant {p_global['mu']}")
assert r_si == lmp.extract_variable('r_si', vartype=LMP_VAR_EQUAL)
rho0 = lmp.extract_variable('rho', vartype=LMP_VAR_EQUAL)
if dmp_interval:
    if 'power' in p_global['restart_file']:  # power-law fluid, output viscosity
        lmp.commands_string("""
        dump	my_dump all custom ${dmp_interval} &
                ${case_path}/${go_on_from_step}to${end_step}.dump &
                id type x y z vx vy vz fx fy fz mass c_p c_rho c_strainAvg v_ringID_paper &
                v_Fy_bath v_Fz_bath v_FRy v_FRz f_F_active[2] f_F_active[3] c_visc proc
        """)
    else:
        lmp.commands_string("""
        dump	my_dump all custom ${dmp_interval} &
                ${case_path}/${go_on_from_step}to${end_step}.dump &
                id type x y z vx vy vz fx fy fz mass c_p c_rho c_strainAvg v_ringID_paper &
                v_Fy_bath v_Fz_bath v_FRy v_FRz f_F_active[2] f_F_active[3] proc
        """)
    lmp.commands_string("""
    # variable tDump equal stride(536200,539750,50)
    dump_modify my_dump format float %.6e
    """)

net_comp_time = None
interact_comp_time = None
t0 = None
rhomin_flag = 0
rhomin = 1000
# control wave interval
strain0_prev = 0
rest0 = False if strain0 < 0 else True
if me == 0:
    net_comp_time = 0.
    interact_comp_time = 0.
    t0 = time()


def callback(caller, step, nlocal, tag, x, fext):
    global net_comp_time, interact_comp_time, rhomin_flag, rhomin, \
        strain0, strain0_prev, rest0, t_since_activated
    # rhomin < rho0 implies fluid particles penetrate out of the wall, should quit
    # if rhomin_flag:
    #     warn(f'rhomin={rhomin:.2f}. Fluid particles are likely to penetrate out of the wall!')
    #     caller.error(LMP_ERROR_ALL, 'rhomin<rho0')

    step_local = step - go_on_from_step
    t0_step = None
    net_comp_time_step = None
    mylog_flag = 0
    if me == 0:
        mylog_flag = step_local % mylog_interval == 0
        t0_step = time()
        net_comp_time_step = 0.
    xyz_all = caller.gather_atoms_subset('x', 1, 3, n_wall, id_wall)
    # obtain strain
    tags = comm.gather(tag - 1, root=0)
    fs = None
    if step_local > 0:
        strain = np.array([caller.numpy.extract_fix(
            'strainAvg', LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY, nrow=nrow, ncol=1
        ) for nrow in range(n_ring)])
        strains.append(strain)
    elif go_on_from_step > 0:
        strain = strains[-1]
    else:
        strain = np.zeros(n_ring)
    strain_flag = (strain.min() < -0.5) or (strain.max() > 0.3)
    # average the strain every two SMC for mechanosensing
    strain_sense = strain[ring_sense_start:ring_sense_end + 1].reshape(-1, n_sense_each).mean(axis=1)
    # # obtain rhomin
    # rhomin = caller.extract_compute('rhomin', LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)
    # rhomin_flag = (rhomin < rho0) and (dmp_interval == 0 or step_local % dmp_interval == 0)
    if me == 0:
        value_info = None
        if mylog_flag:
            strain_id = list(range(0, n_sense, 1))
            strain_head = '\t\t'.join(str(i) for i in strain_id)
            strain_display = '\t'.join(f'{i:<6.1%}' for i in strain_sense[strain_id])
            value_info = (f"strain_max\t\t{strain_sense.argmax()}\t{strain_sense.max():.2%}\n"
                          f"strain_min\t\t{strain_sense.argmin()}\t{strain_sense.min():.2%}\n"
                          f"{strain_head}\n"
                          f"{strain_display}\n")
        if step_local > 0:  # LAMMPS's pre-run won't lead to Brian2 running
            # determine allow_spike
            strain0_prev = strain0
            strain0 = strain_sense[0]
            if strain0 < 0 and strain0 < strain0_prev and rest0:
                rest0 = False
                t_since_activated = 0
            elif strain0 > 0.04 and strain0 > strain0_prev and not rest0:
                rest0 = True
            allow_spike = np.ones(ring_sense_end - ring_sense_start + 1)
            left_wave_ringIDs = np.argwhere(strain[ring_sense_start:ring_sense_end + 1] <= 0)
            if left_wave_ringIDs.size:
                left_wave_ringID = left_wave_ringIDs[0].item()
                allow_spike[:left_wave_ringID] = int(t_since_activated > wave_interval_min)
            allow_spikes.append(allow_spike)
            t_net0 = time()
            if (strain_sense != 0).any():
                net['IPAN'].DSTND = strain_sense
            net['SMC'].allow_spike = allow_spike
            net.run(dt_couple * second)
            t_since_activated += dt_couple
            tensions.append(np.array(net['SMC'].T_avg))
            net_comp_time_step = time() - t_net0
            net_comp_time += net_comp_time_step
        if (step_local % save_interval == 0 and step_local > 0) or strain_flag:
            with open(f'{case_path}/state/net_{step}.pkl', 'wb') as pf:
                pickle.dump(net.get_states(read_only_variables=False), pf)
            np.savez(f'{case_path}/interface/strain_tension_{step}.npz',
                     strain=np.array(strains), tension=np.array(tensions),
                     allow_spikes=np.array(allow_spikes))
            with open(f'{case_path}/restart_params/state_scalars_{step}.yml', 'w') as ymf:
                yaml.safe_dump({'t_since_activated': t_since_activated}, ymf)
        f_mag = np.repeat(net['SMC'].T_avg, n_muscle_each)
        # debug mode ===============
        # f_mag = np.zeros(184)
        # f_mag[-2:] = force_factor*0.01

        # calculate the fy and fz by fmag
        yz_all = np.asarray(xyz_all).reshape(-1, n_yz, 3)[:, :, 1:]  # (184,63,2)
        assert yz_all.shape[0] == n_ring
        yz_all = yz_all[ring_sense_start:ring_sense_end + 1]  # (184,63,2)
        center_to_yz = yz_all.mean(axis=1, keepdims=True) - yz_all  # (184,63,2)
        f_wall = f_mag[:, np.newaxis, np.newaxis] / np.sqrt(
            (center_to_yz ** 2).sum(axis=-1, keepdims=True)) * center_to_yz  # (184,63,2)
        # prepare fs for comm.scatter
        tag_max = np.hstack(tags).max() + 1
        f_all = np.zeros((tag_max, 3))
        f_all[id_sense_start - 1:id_sense_end, 1:] = f_wall.reshape(-1, 2)
        fs = [f_all[tag] for tag in tags]

        if mylog_flag:
            if allow_spikes:
                allow_spike = allow_spikes[-1]
                if allow_spike.all():
                    allow_info = 'All true'
                else:
                    allow_info = f'0-{int(np.sum(1 - allow_spike) - 1)} false'
            else:
                allow_info = 'None'
            step_info = (f"Step (curr -- total): "
                         f"{step_local}/{n_step} -- {step}/{end_step}"
                         f"\n{'=' * 30}")
            value_info = (f'Time since last activation is {t_since_activated: .3f} second\n'
                          f'rest[0]: {bool(rest0)}, allow_spikes: {allow_info}\n') + value_info
            tension_id = list(range(0, n_sense * n_sense_each, n_sense_each))
            tension_head = '\t\t'.join(str(i) for i in tension_id)
            tension_display = '\t'.join(f'{i:<4.1f}' for i in f_mag[tension_id] * 1e6)
            vSMC = np.repeat(net['SMC'].v / mV, n_muscle_each)
            vSMC_display = '\t'.join(f'{i:<5.1f}' for i in vSMC[tension_id])
            value_info += (
                f"{tension_head}\n"
                f"vSMC_max/mV\t\t{(net['SMC'].v / mV).argmax()}\t{(net['SMC'].v / mV).max():.1f}\n"
                f"{vSMC_display}\n"
                f"tension_max/uN\t\t{f_mag.argmax()}\t{f_mag.max() * 1e6:.1f}\n"
                f"{tension_display}\n"
            )
            interact_comp_time += time() - t0_step - net_comp_time_step
            elapsed = time() - t0
            lmp_comp_time = elapsed - net_comp_time - interact_comp_time
            rate_net = (step_local * dt_lmp / dt_net) / (net_comp_time + 0.01)
            rate_lmp = (step_local + 1) / lmp_comp_time
            rate_total = (step_local + 1) / elapsed
            eta = elapsed / (step_local + 1) * (n_step - step_local)
            total_time_estimated = elapsed + eta
            time_info = ('Time elapsed\t' + format_time(elapsed) +
                         '\tMechanics for ' + format_time(lmp_comp_time) +
                         '\tNetwork for ' + format_time(net_comp_time) +
                         '\tInteraction for ' + format_time(interact_comp_time) +
                         '\nETA\t\t\t\t' + format_time(eta) +
                         '\tTotal time estimated\t' + format_time(total_time_estimated) +
                         f'\nComputation rate (step/sec): {rate_total:.1f}\t'
                         f'Mechanics: {rate_lmp:.1f}\t'
                         f'Network: {rate_net:.1f}')
            with open(f'{case_path}/my{go_on_from_step}_to{end_step}.log', 'a') as f:
                print(f'{step_info}\n{value_info}\n{time_info}\n\n', file=f, flush=True)
    if strain_flag:
        warn('The strain is too negative/positive!')
        caller.error(LMP_ERROR_ALL, 'The strain is too negative/positive!')
    fext[:] = comm.scatter(fs, root=0)
    comm.Barrier()


lmp.set_fix_external_callback('F_active', callback, lmp)
if balance == 'rcb':
    lmp.commands_string("""
    comm_style tiled
    fix bl all balance 100000 1.01 rcb
    """)
elif balance == 'shift':
    lmp.command('fix bl all balance 100000 1.01 shift x 20 1.01')
else:
    lmp.command('fix bl all balance 100000 1.01 report')
lmp.commands_string("""
thermo	        20
thermo_style	custom step time press atoms c_rhomin c_rhomax &
                f_in_del f_out_del f_in_depos f_out_depos f_bl[3] f_bl
thermo_modify	norm no lost/bond warn
""")
lmp.command(f"run {n_step}")
lmp.command(f'write_restart {case_path}/restart/*.restart')
if me == 0:
    with open(f'{case_path}/state/net_{end_step}.pkl', 'wb') as pf:
        pickle.dump(net.get_states(read_only_variables=False), pf)
    np.savez(f'{case_path}/interface/strain_tension_{end_step}.npz',
             strain=np.array(strains), tension=np.array(tensions),
             allow_spikes=np.array(allow_spikes))
