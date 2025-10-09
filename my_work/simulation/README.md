# Configure the environment variables

Ensure `LMP_PLUGIN_PATH_RELEASE` has been included in the environment variables as `../../LAMMPS-src/README.md`, which is used in `head-rheo.in`.

Add the absolute path of `../` (`my_work`) to the environment variable as `MY_WORK`, which will be used in `head-rheo.in`, `bio_intestine_rheo.in`, and `neuron_fluid_brian_rheo.py`.

```bash
echo "MY_WORK=path/to/my_work" >> ~/.bashrc
source ~/.bashrc
```

If you don’t want to add any environment variables, you can alternatively replace them in the scripts with the corresponding paths, though we think the former is a bit more convenient.

# Configure the simulation parameters

The simulation parameters are configured in the YAML file in `../results` with the following parameters.

- `case_name`: must be the same as the filename.

## Running

- `n_step`: How many time steps the simulation runs for.
- `go_on_from_step`: Go on from this step. It is useful for diagnostics and continuing computation from unexpected interrupt. For simulation from scratch, this should be `0`.

## Output

- `save_interval`: Save the LAMMPS restart file, Brian2 state file, and the NumPy binary file which stores the strains and active force (tension) every this many time steps.

- `dmp_interval`: Save the LAMMPS dump file every this many time step.

- `mylog_interval`: Update my simulation log (different from LAMMPS log) every this many time step.

## Coupling

- `force_factor`: Maximal active force with a unit of newton.

- `N_callback_lmp`: The mechanical module is to communicate with the physiological module every this many time step to reduce the overhead on repeated restart of computation.

## Mechanics

- `restart_file`: LAMMPS restart file. It is effective only when `go_on_from_step == 0`.
- `dt_lmp`: time step of the mechanical module with a unit of second.
- `soft`: Prefactor of the soft potential with a unit of joule.
- `mu`: Dynamic viscosity of the intestinal content with a unit of Pa*s.
- `kB_circ`: Stiffness of the circumferential spring with a unit of N/m.
- `kB_long`: Stiffness of the longitudinal spring with a unit of N/m.
- `k1_self`: Prefactor of the linear term of the restoring force with a unit of N/m.
- `k2_self`: Prefactor of the quadratic term of the restoring force with a unit of N/m^2^.
- `kA`: Prefactor of the harmonic angle potential with a unit of joule.
- `balance`: Whether to turn on load balancing. Can be `yes/on/true/no/off/false`.

## Physiology

- `n_sense_each`: Every this many neurons correspond to one SMC.

- `ring_sense_start`: RingID of IPAN #0.

- `JPalpha`: Parameter of the SMC electrophysiological model with a unit of mV.

- `V_half_EJP`: Parameter of the SMC electrophysiological model with a unit of mV.

- `V_half_IJP`: Parameter of the SMC electrophysiological model with a unit of mV.

- `w_boundary1`: Parameter of the SMC (#0, #1, #N-2, #N-1) electrophysiological model with a unit of mV to consider the boundary effect.

- `w_boundary2`: Parameter of the SMC (#2, #3, #N-4, #N-3) electrophysiological model with a unit of mV to consider the boundary effect.