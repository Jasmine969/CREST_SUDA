# CREST_SUDA

The small intestine propels contents forward via peristalsis, which is controlled by a closed-loop system: a contraction upstream pushes the contents downstream and stretches the wall ahead; stretch-sensing nerves detect this signal and correspondingly regulate muscle activity, shaping the next wave.

We built a computational framework, CREST, that implements this feedback loop under neural control. Peristaltic waves could be produced with velocities and strengths matching experimental data. We also numerically discovered that the peristaltic wave could travel in a series of leaps. In addition to understanding normal intestinal functions, CREST can potentially help researchers test mechanistic hypotheses about motility disorders.

<img width="4452" height="4365" alt="模型框架 - neurogenic" src="https://github.com/user-attachments/assets/2ff8010f-5fce-4707-afda-c5fe05b53c48" />


This repository provides the source code of the paper (in press): 
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
