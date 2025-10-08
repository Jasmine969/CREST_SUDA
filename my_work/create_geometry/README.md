To create the geometry, run `create_all_rheo-atoms_bpm-bonds2_angles.py`. The particles constitute an inlet, an outlet, the SI wall, and fluid. `bpm/spring` bonds with the angle constraint are also included.

Then we should perfuse the fluid into the lumen until equilibrium. To achieve this, run `inlet_si_outlet_bond-rheo-inflow.in`:

```bash
mpirun -np [number of processors] lmp -in inlet_si_outlet_bond-rheo-inflow.in
```

Some fluid particles have been created by `create_all_rheo-atoms_bpm-bonds2_angles.py` to accelerate inflow. One can first run 2000 steps to equilibrate these fluid particles. From the 2001st step, turn on fluid deposition and evaporation. Run the simulation until the particle count does not change. Save the restart file for ensuing simulation.
