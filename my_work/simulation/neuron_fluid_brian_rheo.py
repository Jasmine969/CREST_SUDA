import os
import sys
from socket import gethostname

import yaml
from warnings import warn
import pickle
from mpi4py import MPI
from lammps import (lammps, LMP_VAR_EQUAL, LMP_ERROR_ALL,
                    LMP_STYLE_GLOBAL, LMP_TYPE_ARRAY, c_int)
from brian2 import numpy_ as np
from brian2.units import second, mV
from time import time, asctime
from pathlib import Path

root_path = Path(os.getenv('MY_WORK')).parent
sys.path.append(str(root_path))
from my_work.network_models.Motility_Model_noICC import MotilityModel
from utils.utils import format_time

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
# assert dmp_interval % N_callback_lmp == 0
dt_couple = dt_lmp * N_callback_lmp
N_callback_net = int(dt_couple / dt_net)
assert N_callback_net * dt_net == dt_couple
n_ring = int(n_wall / n_yz)
assert n_wall == n_yz * n_ring
n_sense_each = p_global['n_sense_each']
ring_sense_start = p_global['ring_sense_start']  # 9
ring_sense_end = n_ring - (ring_sense_start - 1)  # 192
n_sense = int((ring_sense_end + 1 - ring_sense_start) / n_sense_each)  # 92
n_muscle_each = int(n_sense_each / 2)
assert n_muscle_each * 2 == n_sense_each
gr_sense_start = int(ring_sense_start / n_sense_each) + 1  # 5
gr_sense_end = int(ring_sense_end / n_sense_each)  # 96
id_sense_start = int(n_inlet + (ring_sense_start - 1) * n_yz + 1)
id_sense_end = int(n_inlet + ring_sense_end * n_yz)
id_wall = np.arange(n_inlet, n_inlet + n_wall) + 1
assert id_wall[n_yz * (ring_sense_start - 1)] == id_sense_start
assert id_wall[n_yz * ring_sense_end - 1] == id_sense_end
id_wall = (n_wall * c_int)(*id_wall)
peri0 = 2 * np.pi * r_si
# network params
net_params = None
if me == 0:
    net_params = {
        'neuron_pop': n_sense,
        'force_factor': force_factor,
        'JPalpha': p_global['JPalpha'],
        'V_half_EJP': p_global['V_half_EJP'],
        'V_half_IJP': p_global['V_half_IJP'],
        'w_boundary1': p_global['w_boundary1'],
        'w_boundary2': p_global['w_boundary2'],
        # 'G_SAC': p_global['G_SAC'],
        'N_callback_net': N_callback_net,
        'dt_couple': dt_couple,
        'dt_net': dt_net,
    }
    if 'w_cGMP_ICC' in p_global:
        net_params.update({
            'w_cGMP_ICC': p_global['w_cGMP_ICC']
        })

# go_on_from_step==0 if ab initio
go_on_from_step = p_global['go_on_from_step']
n_step = p_global['n_step']
end_step = go_on_from_step + n_step
if me == 0:
    for directory in ['restart', 'state', 'interface']:
        os.makedirs(f'{case_path}/{directory}', exist_ok=True)
    print('See my.log')
    with open(f'{case_path}/my{go_on_from_step}_to{end_step}.log', 'w') as f:
        print(f'case: {case_name}\nRunning on {gethostname()}'
              f'\n{asctime()}\n\n', file=f)
if me == 0 and n_step // save_interval > 200:
    warn('There will be over 200 result files.'
         ' Change save_interval or continue? (y/n)\n')
    if_continue = input()
    if if_continue.lower() in ['n', 'no', 'false']:
        raise Exception
# multiphysics
lmp = lammps(cmdargs=['-screen', 'none'])
# lmp = lammps()
lmp.commands_string(f"""
log      {case_path}/log_{go_on_from_step}to{end_step}.lammps
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
if comm.size > 1 and not p_global['balance']:
    lmp.command('processors * 2 1')
# neurons
net = None
if me == 0:
    net = MotilityModel(go_on_from_step, **net_params)
if go_on_from_step:
    lmp.command(f'read_restart {case_path}/restart/{go_on_from_step}.restart')
    if me == 0:
        with open(f'{case_path}/state/net_{go_on_from_step}.pkl', 'rb') as pf_state:
            states = pickle.load(pf_state)
            # must reset lastspike so that IPAN will not be refractory
            for objects in states.values():
                for k, val in objects.items():
                    if k.startswith("lastspike"):
                        val -= 1000 * second
            net.set_states(states)
    st = np.load(f'{case_path}/interface/strain_tension_{go_on_from_step}.npz')
    strains = list(st['strain'])
    tensions = list(st['tension'])
else:
    lmp.commands_string(f"""
    read_restart   ../create_geometry/data/{p_global['restart_file']}
    reset_timestep 0
    """)
    strains = []
    tensions = []
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
                v_Fy_bath v_Fz_bath v_FRy v_FRz f_F_active[2] f_F_active[3] c_visc
        """)
    else:
        lmp.commands_string("""
        dump	my_dump all custom ${dmp_interval} &
                ${case_path}/${go_on_from_step}to${end_step}.dump &
                id type x y z vx vy vz fx fy fz mass c_p c_rho c_strainAvg v_ringID_paper &
                v_Fy_bath v_Fz_bath v_FRy v_FRz f_F_active[2] f_F_active[3]
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
if me == 0:
    net_comp_time = 0.
    interact_comp_time = 0.
    t0 = time()


def callback(caller, step, nlocal, tag, x, fext):
    global net_comp_time, interact_comp_time, rhomin_flag, rhomin
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
    strain_flag = (strain.min() < -0.5) or (strain.max() > 0.25)
    # average the strain every two SMC for mechanosensing
    strain_sense = strain[ring_sense_start - 1:ring_sense_end].reshape(-1, 2).mean(axis=1)
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
            # update distension and run brian2
            t_net0 = time()
            if (strain_sense != 0).any():
                net['IPAN'].DSTND = strain_sense
            net.run(dt_couple * second)
            tensions.append(np.array(net['SMC'].T_avg))
            net_comp_time_step = time() - t_net0
            net_comp_time += net_comp_time_step
        if (step_local % save_interval == 0 and step_local > 0) or strain_flag:
            with open(f'{case_path}/state/net_{step}.pkl', 'wb') as pf:
                pickle.dump(net.get_states(read_only_variables=False), pf)
            np.savez(f'{case_path}/interface/strain_tension_{step}.npz',
                     strain=np.array(strains), tension=np.array(tensions))
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
            step_info = (f"Step (curr -- total): "
                         f"{step_local}/{n_step} -- {step}/{end_step}"
                         f"\n{'=' * 30}")
            tension_id = list(range(0, n_sense * 2, 2))
            tension_head = '\t\t'.join(str(i) for i in tension_id)
            tension_display = '\t'.join(f'{i:<4.1f}' for i in f_mag[tension_id] * 1e6)
            vSMC = np.repeat(net['SMC'].v / mV, n_muscle_each)
            vSMC_display = '\t'.join(f'{i:<5.1f}' for i in vSMC[tension_id])
            # vICC = np.repeat(net['ICC'].v / mV, n_muscle_each)
            # vICC_display = '\t'.join(f'{i:<5.1f}' for i in vICC[tension_list])
            value_info += (
                # f"vICC_max\t\t{(net['ICC'].v / mV).argmax()}\t{(net['ICC'].v / mV).max():.1f}\n"
                f"{tension_head}\n"
                # f"{vICC_display}\n"
                f"vSMC_max/mV\t\t{(net['SMC'].v / mV).argmax()}\t{(net['SMC'].v / mV).max():.1f}\n"
                f"{vSMC_display}\n"
                f"tension_max (uN)\t\t{f_mag.argmax()}\t{f_mag.max() * 1e6:.1f}\n"
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
                print(f'{step_info}\n{value_info}\n{time_info}\n\n', file=f)
    if strain_flag:
        warn('The strain is too negative/positive!')
        caller.error(LMP_ERROR_ALL, 'The strain is too negative/positive!')
    fext[:] = comm.scatter(fs, root=0)
    comm.Barrier()


lmp.set_fix_external_callback('F_active', callback, lmp)
if comm.size > 1 and p_global['balance']:
    lmp.commands_string("""
        comm_style tiled
        fix bl all balance 100000 1.01 rcb
    """)
lmp.command(f"run {n_step}")
lmp.command(f'write_restart {case_path}/restart/*.restart')
if me == 0:
    with open(f'{case_path}/state/net_{end_step}.pkl', 'wb') as pf:
        pickle.dump(net.get_states(read_only_variables=False), pf)
    np.savez(f'{case_path}/interface/strain_tension_{end_step}.npz',
             strain=np.array(strains), tension=np.array(tensions))
