# CREST_SUDA
Source code of the paper (in press): 
**Hong Zhu**, Meng Wai Woo, Xiao Dong Chen & Jie Xiao. Closed-loop control of the small intestinal transport within an integrated mechano-physiological framework. *J. R. Soc. Interface*. 2026. 10.1098/rsif.2025.1155

# Structure

```
├─LAMMPS-src  			// LAMMPS source directory
│  ├─cmake 				// Customized cmake file
│  ├─src 				// Modified LAMMPS source codes
│  └─src_plugins 		// New LAMMPS plugins
├─my_work 				// Workflow
│  ├─create_geometry 	// LAMMPS geometry construction and equilibration
│  ├─network_models 	// Models of various neurons and synapses
│  ├─simulation			// Simulation scripts
│  ├─postprocess 		// Postprocessing scripts
│  └─results 			// Configuration files and corresponding results
└─utils 				// Utility functions
```

# Usage

1. Create a Python virtual environment and then install the required Python packages.
2. Compile the LAMMPS source code. See `LAMMPS-src/README.md` for details.
3. Create the geometry and equilibrate the particles by the scripts in `my_work/create_geometry/`.
4. Customize your configuration file in `my_work/results/` and run the simulation by `my_work/simulation/neuron_fluid_brian_rheo.py`.
5. Postprocess the results by the scripts in `my_work/postprocess`.
