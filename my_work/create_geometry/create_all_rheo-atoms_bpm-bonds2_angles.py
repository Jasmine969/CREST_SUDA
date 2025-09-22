"""
Create the inlet (vertical + torus pipe), SI wall, and fluid.
Bonds bpm/spring.
The bond is classified into circumferential (1) and longitudinal (2)
"""
from my_geometry import *
from lammps import lammps
import numpy as np
from tqdm import trange
from functools import partial
from utils.geometry import get_wall_ID

out = ''
lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
rho_fluid = 993
rho_wall = 1040
l_pipe_hrz = 0.002
l_pipe_vert = 0.004
r_si = 0.002
# r_catheter==r_si -> original
r_catheter = 0.002
thick_si = 3e-4
r_torus = 2 * r_catheter
dl = 2e-4
l_si = 0.04 - dl

pipe_in_vert = CylinderSide(
    r=r_catheter, l_axis=l_pipe_vert, dl=dl, axis='z'
).shift(z=r_torus)
# l_pipe may change when creating the cylinder,
# so x-shift it after creation
l_pipe_vert = pipe_in_vert.l_axis
n_ring_catheter = pipe_in_vert.n_ring
pipe_in_vert = pipe_in_vert.shift(x=-r_torus - l_pipe_hrz)
log_n_ring = f'{n_ring_catheter} particles per catheter-ring.'
torus_in = Torus(
    r_ring=r_catheter, r_t=r_torus, dl=dl, n_ring=n_ring_catheter,
    phi_range='(180,270)', plane='XOZ'
).shift(z=r_torus, x=-l_pipe_hrz)
if r_catheter == r_si:
    lid_in = Geometry()
    pipe_in_hrz = CylinderSide(
        r=r_si, l_axis=l_pipe_hrz - dl, dl=dl, axis='x',
    ).shift(x=-l_pipe_hrz)
else:
    lid_in = ThickRing(r_out=r_si, r_in=r_catheter, dl=dl,
                       incl_outer=True, incl_inner=True, axis='x', adjust_dl=True
                       ).shift(x=-l_pipe_hrz)
    pipe_in_hrz = CylinderSide(
        r=r_si, l_axis=l_pipe_hrz - dl * 2, dl=dl, axis='x',
    ).shift(x=-l_pipe_hrz + dl)
n_ring_si = pipe_in_hrz.n_ring
log_n_ring += f' {n_ring_si} particles per SI-ring.'
print(log_n_ring)
inlet = Union(pipe_in_vert, torus_in, lid_in, pipe_in_hrz)
si = CylinderSide(r=r_si, l_axis=l_si, dl=dl, axis='x')
log_si = f'dl in a ring is {si.dl_in_ring}. SI has {si.n_axis} rings.'
print(log_si)
l_si = si.l_axis

outlet = inlet.mirror(plane_name='YOZ', plane_pos=l_si / 2)

r_fluid1 = r_si
n_axis_fluid1 = int((l_si + 2 * l_pipe_hrz - 2 * dl) / dl + 1)
l_fluid1 = (n_axis_fluid1 - 1) * dl
fluid1 = Stack(
    ThickRing(r_out=r_fluid1, r_in=0, dl=dl, incl_inner=True, incl_outer=False, axis='x'),
    'x', n_axis_fluid1, dl
).shift(x=-l_pipe_hrz + dl)
r_fluid2 = r_catheter
n_axis_fluid2 = int(l_pipe_vert*0.9 / dl + 1)
l_fluid2 = (n_axis_fluid2 - 1) * dl
fluid2 = Stack(
    ThickRing(r_out=r_fluid2, r_in=0, dl=dl, incl_inner=True, incl_outer=False, axis='z'),
    'z', n_axis_fluid2, dl
).shift(x=-l_pipe_hrz - r_torus, z=r_torus)
fluid3 = fluid2.mirror(plane_name='YOZ', plane_pos=l_si / 2)
fluid = Union(fluid1, fluid2, fluid3)

buffer_region = 0.1 * l_si
xlo = -(l_pipe_hrz + r_torus + r_si) - buffer_region
xhi = l_si + l_pipe_hrz + r_torus + r_si + buffer_region
ylo = -r_si - buffer_region
yhi = r_si + buffer_region
zlo = -r_si - buffer_region
zhi = l_pipe_vert + r_torus + buffer_region
n_bond_type = 2
n_bond_per_atom = 4
lmp.commands_string(f"""
dimension	3
atom_style	hybrid bond angle rheo
units		si
newton	 	on
boundary	f f f
comm_modify vel yes
region      simulation_box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi} 
create_box  3 simulation_box bond/types {n_bond_type} extra/bond/per/atom {n_bond_per_atom} &
            angle/types 1 extra/angle/per/atom 3
""")
n_atoms_inlet = inlet.size
lmp.create_atoms(
    n_atoms_inlet, np.arange(n_atoms_inlet) + 1 + lmp.get_natoms(),
    np.ones(n_atoms_inlet, dtype=int), inlet.flatten_coords)
n_atoms_si = si.size
lmp.create_atoms(
    n_atoms_si, np.arange(n_atoms_si) + 1 + lmp.get_natoms(),
    np.full(n_atoms_si, 2, dtype=int), si.flatten_coords)
n_atoms_outlet = outlet.size
lmp.create_atoms(
    n_atoms_outlet, np.arange(n_atoms_outlet) + 1 + lmp.get_natoms(),
    np.ones(n_atoms_outlet, dtype=int), outlet.flatten_coords)
n_atoms_fluid = fluid.size
lmp.create_atoms(
    n_atoms_fluid, np.arange(n_atoms_fluid) + 1 + lmp.get_natoms(),
    np.full(n_atoms_fluid, 3, dtype=int), fluid.flatten_coords)
n_atoms_all = lmp.get_natoms()
log_n_atom = f'n_atoms_inlet: {n_atoms_inlet},' \
             f' n_atoms_si: {n_atoms_si},' \
             f' n_atoms_outlet: {n_atoms_outlet},' \
             f' n_atoms_fluid: {n_atoms_fluid},' \
             f' n_atoms_all: {n_atoms_all}.'
print(log_n_atom)

m0_fluid = np.pi * (r_fluid1 ** 2 * l_fluid1 + r_fluid2 ** 2 * l_fluid2 * 2) * rho_fluid / n_atoms_fluid
m0_wall = np.pi * (2 * r_si - thick_si) * thick_si * l_si * rho_wall / n_atoms_si
lmp.commands_string(f"""
mass            * {m0_wall}
mass            3 {m0_fluid}
set group all rheo/rho {rho_fluid}
""")
# =============== create bonds
lmp.commands_string(f"""
group           wall type 2
group           equip type 1
group           equipwall union equip wall
pair_style      zero {dl * 2}
pair_coeff      * *
neighbor        {dl * 0.1} bin
create_bonds many wall equipwall 1 0 0.0001995
create_bonds many wall equipwall 2 0.0001996 0.00021
""")
n_bond_nearest = si.size + si.n_ring * (si.n_axis + 1)
n_bond_expected = n_bond_nearest
n_bond_created, _ = lmp.gather_bonds()

print('Connect per ring: angles')
get_wall_ID = partial(get_wall_ID, n_per_ring=n_ring_si, smallest_ID=n_atoms_inlet + 1)
for i in trange(1, n_ring_si + 1):
    for j in range(1, si.n_axis + 1):
        lmp.command(f'create_bonds single/angle 1 '
                    f'{get_wall_ID(i, j)} {get_wall_ID(i + 1, j)} {get_wall_ID(i + 2, j)}')
n_angle_expected = n_ring_si * si.n_axis
n_angle_created, _ = lmp.gather_angles()

filename = f'equip_si-rheo_bond2_angle_dl{dl:.1e}-semifilled'
lmp.commands_string(f"""
run             0
write_restart   data/{filename}.restart
write_data      data/{filename}.data
""")
if n_bond_created != n_bond_expected:
    raise UserWarning(f'n_bond_created={n_bond_created},'
                      f' but n_bond_expected={n_bond_expected}.')
else:
    print(f'{n_bond_expected} bonds are created successfully!')
if n_angle_created != n_angle_expected:
    raise UserWarning(f'n_angle_created={n_angle_created},'
                      f' but n_angle_expected={n_angle_expected}.')
else:
    print(f'{n_angle_expected} angles are created successfully!')
# check atom overlapping
lmp.commands_string(f"""
delete_atoms    overlap {dl * 0.8} all all
""")
n_atoms_all_now = lmp.get_natoms()
if n_atoms_all == n_atoms_all_now:
    print('Congrats, no atoms are overlapped!')
else:
    raise RuntimeError(f'{n_atoms_all - n_atoms_all_now} atoms are overlapped!')

with open(f'data/{filename}.meta', 'w') as file_out:
    file_out.write(
        f'{log_n_ring}\nPipe length: {pipe_in_vert.log}\n'
        f'{log_si}\nSI wall: {si.log}\n{log_n_atom}\n'
        f'r_torus: {r_torus}')
