# CREST_SUDA
Source code of the paper: 
Closing the feedback loop: CREST, an integrated mechano-physiological framework for intestinal transport control

# Structure

```
├─LAMMPS-src  			// LAMMPS source directory
│  ├─cmake 				// Customized cmake file
│  ├─src 				// Modified LAMMPS source codes
│  └─src_plugins 		// New LAMMPS plugins
├─my_work 				// Workflow
│  ├─create_geometry 	// LAMMPS geometry construction and equilibration
│  ├─network_models 	// Models of various neurons and synapses
│  ├─postprocess 		// Postprocessing scripts
│  └─results 			// Configuration files and corresponding results
└─utils 				// Utility functions
```

# Usage

1. Create a Python virtual environment and then install the required Python packages.
2. Compile the LAMMPS source code. See `LAMMPS-src/README.md` for details.
3. Create the geometry and equilibrate the particles by the scripts in `my_work/create_geometry/`.
4. Customize your configuration file in `my_work/results/` and run the simulation by `my_work/neuron_fluid_brian_rheo.py`.
5. Postprocess the results by the scripts in `my_work/postprocess`.
