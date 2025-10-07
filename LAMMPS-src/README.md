The LAMMPS source code has been modified and supplemented with some plugins. Users must compile LAMMPS to use them.

# Introduction to modifications

`// jsm` can be seen in the source code to mark the modifications in comparison with the original version.

## Modification of fix rheo
To fix the solid particles stationary, the official example use `fix 5 rig setforce 0.0 0.0 0.0` (lammps-2Apr2025/examples/rheo/dam-break/in.rheo.dam.break). This will eliminate the force exerted on the solid, which is bad for diagnostics. To resolve this, we introduce a new keyword `fixv/type` whose args are boolean values `fixv1, fixv2, ..., fixvN` to specify whether the velocity of a certain type of particle should be fixed (1) or updated (0). In our simulation, there are three types of particles: equipment (inlet and outlet, type 1), intestinal wall (type 2), and fluid (type 3). The equipment is fixed and the other two can move, so we can use:

```
fix rheo all rheo ${cut} quintic 0 density ${rho} ${rho} ${rho} speed/sound $c $c $c &
    fixed/vel 1 0 0
```

The code (`src/RHEO/fix_rheo.cpp`) was not written as a plugin because source codes of the RHEO package relies on each other.

## Two more attributes accessible to users

To compute the total SPH or bond force exerted on a particle, users can now use the command `compute rheo/property/atom` with attributes `fsph` or `fbpm`. The syntax is the same as `shift/v/a`:

```
compute 1 rheo/property/atom fsph/*
compute 2 rheo/property/atom fbpm/y fbpm/z
```



## Plugin `fix ave/chunk/preforce`

The averaged wall strains are requested in `fix external` during `post_force`. To average the wall strain, we need to first compute the centroid of each ring and then do the average by `fix ave/chunk`. However, `fix ave/chunk` is executed during `end_of_step` before `post_force`. To ensure the centroids are available during `fix external`, we modify `fix_ave_chunk.cpp` by changing the mask `END_OF_STEP` to `PRE_FORCE`. The style name is changed to `fix ave/chunk/preforce`. Other keywords and arguments are kept the same.

## Plugin `angle_style harmonic/concave`

The angle constraint should be included for the bonded ring to preserve the round shape. The range of the angle in the original version is $[0,\pi)$, which can lead to a sinking particle with small angle potential. To resolve this, we adjust the range to $[0,2\pi)$ so that concave angles can also be detected. Three more arguments should be specified compared with `angle_style harmonic`:

- equilibrium angle with a degree unit;
- the axis to which the ring is perpendicular (only 2D angles can be processed for the time being so this argument is necessary). It can be `x`, `y` or `z`;
- how many particles the ring has.

Example:

```
angle_style harmonic/concave
angle_coeff 1 ${kA} ${theta_eq} x ${n_yz}
```

The code (in `src_plugins/MOLECULE/angle_harmonic_concave.cpp`) was modified based on `angle_harmonic.cpp`.

## Plugin `fix rheo/deposit`

`fix deposit` is needed to implement the inflow boundary. However, users cannot specify the density (`rheo/rho`) and status (`rheo/status`) in the original version. So we write this plugin with two more optional keywords compared to `fix deposit`:

- `rho`. The argument is the density. Default is 1000.
- `status`. The argument is the status of the particle (0 for fluid and 1 for solid). Default is 0.

Example:

```
fix depos fluid sph/deposit 130000 3 ${Ninflow} 623 region pipe_in_vert near ${dL} &
    vx 0 0 vy 0 0 vz 0 0 rho ${rho} status 0
```

The code (`src_plugins/RHEO/fix_rheo_deposit.cpp`) was modified based on `fix_deposit.cpp`.

## Plugin `pair_style sph/rhosum/norm`

This is an implementation of density renormalization. One more optional keyword can be specified compared to `pair_style sph/rhosum`:

- `zmin/rho`. The arguments are minimum coordination number `zmin` and the initial density `rho0`. When the coordination number of one particle is smaller than `zmin`, its density will be set as `rho0`.

Example:

```
pair_style sph/rhosum/norm 2
pair_coeff ${cut} zmin/rho0 5 ${rho}
```

The code (`src_plugins/SPH/pair_sph_rhosum_norm.cpp`) was modified based on `pair_sph_rhosum.cpp`.

# Installation

## Preparing for compilation
First, we should prepare the LAMMPS source code. Assume it is in `../../lammps-2Apr2025`. Then let's copy the files in the current directory into LAMMPS directory.

```bash
cp cmake/presets/custom.cmake ../../lammps-2Apr2025/cmake/presets/custom.cmake
cp src/RHEO/* ../../lammps-2Apr2025/src/RHEO/
cp src_plugins -r ../../lammps-2Apr2025/
```

Users have to install MPICH or OpenMPI themselves to compute in parallel. The serial mode can be rather slower.

The BPM package relies on GSL, so the latter should be installed first.

## Compiling the source code
Change to the LAMMPS source directory and compile the source code.
```bash
cd ../../lammps-2Apr2025
mkdir build
cd build
mkdir /opt/lammps
cmake -C ../cmake/presets/custom.cmake \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_Fortran_COMPILER=mpifort \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/lammps \
    -DBUILD_SHARED_LIBS=yes \
    -DPython_EXECUTABLE=path/to/your/python/executable \
    -DGSL_ROOT_DIR=path/to/your/gsl/dir
    ../cmake
make -j
make install
```

Check whether LAMMPS has been installed successfully:
```bash
mpirun -np 4 lmp

LAMMPS (2 Apr 2025)
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread.
  using 1 OpenMP thread(s) per MPI task
```


## Install Python LAMMPS Module
We recommend that users install Python from the [official website](https://www.python.org/) and manage packages with venv, rather than using Anaconda, which may sometimes cause unexpected errors. Activate your Python virtual environment, and then (at the directory `build`)
```bash
make install-python
pip install lammps-2025.7.22xxxxxx.whl
pip install mpi4py
```

Run the following Python script to check whether Python LAMMPS Module has been installed successfully:
```python
from lammps import lammps
from mpi4py import MPI

lmp = lammps()
```

## Install LAMMPS plugins
```bash
cd src_plugins # in the LAMMPS directory
cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Release ../
make -j
```

Add the current path to the environment variable as `LMP_PLUGIN_PATH_RELEASE`, which will be used in `my_work/head-rheo.in`.

```bash
echo "LMP_PLUGIN_PATH_RELEASE=path/to/plugin" >> ~/.bashrc
source ~/.bashrc
```

