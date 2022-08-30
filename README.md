
![FOR2895Logo](for2895_logo.png)

# OpenFOAM simulations and modal analysis of transonic shock buffets

The simulation setups and analysis tools available in this repository were created as part of the research program [FOR 2895](https://www.for2895.uni-stuttgart.de/) 

> Unsteady flow interaction phenomena at high speed stall conditions

financed by the German Research Foundation (DFG). The primary intention of this repository is to provide a fully reproducible workflow for the simulation and modal analysis of transonic shock buffets.

## Results sneak peek

The video below shows a slice of the local Mach number field. The flow conditions are:

- $Re=U_\infty c/\nu = 10^7$
- $Ma_\infty = U_\infty/a_\infty = 0.75$
- $\alpha = 4^\circ$

The simulation setup successfully yields the transonic buffet phenomenon.

https://user-images.githubusercontent.com/8482575/187466191-15d3a8ce-62af-4b9e-a46d-13a4eb49eb03.mp4

The buffet phenomenon is characterized by the complex interaction of shock motion, boundary layer separation, and upstream-propagating acoustic waves. The video below shows the reconstruction of a single DMD mode, which encodes vortex shedding and acoustic waves propagating upstream. The acoustic waves originate in the shear layer and at the trailing edge.

https://user-images.githubusercontent.com/8482575/187465648-08c509c9-889a-4a1e-a68c-1d1790e2b924.mp4

Identifying vortex shedding, acoustic waves, and shock motion requires a robust DMD workflow, which is established by extensive testing of various DMD variants and state vectors. For more details refer to the articles listed under *References*.

## Data-processing and modal analysis

### Getting the data

Due to the size of the full simulations, we provide only processed data and final outputs. If additional data are required, feel free to get in touch, e.g., by opening a new issue. The notebooks expect the data to be located under *notebooks/output/*. To download and place the data correctly, run:
```
# assuming you are at the repository's top level
wget LINK_TO_ARCHIVE
tar xzf results.tar.gz
```

### Python environment

The main library for processing the OpenFOAM data and performing modal decomposition is [flowTorch](https://github.com/FlowModelingControl/flowtorch). FlowTorch and additional dependencies can be installed using *pip*:

```
pip3 install git+https://github.com/FlowModelingControl/flowtorch@aweiner
```

### Scripts and notebooks

The following list of notebooks and scripts might be helpful for finding a particular type of analysis. If the notebooks are not displayed correctly in the browser, clone the repository and open the notebooks locally.

- [naca0012_dmd_rank_spectra.ipynb](notebooks/naca0012_dmd_rank_spectra.ipynb): influence of rank truncation on the DMD spectrum
- [naca0012_dmd_spectogram.ipynb](notebooks/naca0012_dmd_spectogram.ipynb): influence of the number of buffet cycles in combination with small changes in the data and rank truncation on the DMD spectrum
- [naca0012_dmd_variants.ipynb](notebooks/naca0012_dmd_variants.ipynb): comparison of DMD variants in terms of eigenvalues and dominant frequency for various state vectors; requires running *test_dmd_\*.py* scripts first
- [naca0012_animate_dmd_modes.ipynb](notebooks/naca0012_animate_dmd_modes.ipynb): partial reconstruction and animation of selected DMD modes
- [naca0012_lift_and_drag.ipynb](notebooks/naca0012_lift_and_drag.ipynb): visual inspection of lift and drag coefficients of 2D and 3D simulations; frequency analysis
- [naca0012_pressure_coefficients.ipynb](notebooks/naca0012_pressure_coefficients.ipynb): analysis of surface pressure coefficients; mesh dependency study
- [naca0012_probe_analysis.ipynb](notebooks/naca0012_probe_analysis.ipynb): frequency analysis of flow speed at selected probe locations
- [naca0012_schlieren_analysis.ipynb](notebooks/naca0012_schlieren_analysis.ipynb): spatio-temporal correlation coefficients of a line-sample extracted from numerical Schlieren images
- [naca0012_slice_modal_analysis.ipynb](notebooks/naca0012_slice_modal_analysis.ipynb): modal analysis of slice data; spatio-temporal correlation coefficients of line-sample
- [naca0012_state_vectors.ipynb](notebooks/naca0012_state_vectors.ipynb): norms of different state vectors
- [naca0012_surface_modal_analysis.ipynb](notebooks/naca0012_surface_modal_analysis.ipynb): modal analysis of surface pressure coefficients
- [naca0012_volume_modal_analysis.ipynb](notebooks/naca0012_volume_modal_analysis.ipynb): visualization of the modal decomposition computed with *naca0012_volume_modal_analysis.py*
- *extract_lift_and_drag.py*: combines, cleans, and stores the force coefficient data of an OpenFOAM simulation
- *utils.py*: helper functions for loading, processing, analysis, and plotting
- *convert_naca0012_simulations.py*: converts a reconstructed OpenFOAM simulation into flowTorch HDF5 format
- *create_naca0012_slice_dm.py*: extracts a slice from an OpenFOAM simulation and stores the snapshots in a data matrix
- *create_naca0012_surface_dm.py*: extracts the pressure coefficient on the airfoil's upper surface and stores the snapshot data in data matrices

## Performing simulations

All simulations can be executed using Singularity or a local installation of OpenFOAM-v2012. Instructions and dependencies for each workflow follow below.

### Singularity (recommended)

#### Getting the OpenFOAM Singularity image

[Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) is a container tool that allows making results reproducible and performing simulations, to a large extent, platform independent. The only remaining dependencies are Singularity itself and OpenMPI (see next section for further comments). To build the image, run:

```
sudo singularity build of_v2012.sif docker://andreweiner/of_pytorch:of2012-py1.7.1-cpu
```

The resulting image file *of_v2012.sif* should be located at the repository's top level.

#### Running a simulation

When executing an application in parallel using the Singularity image,
it is important that the host's OpenMPI version (installed on your workstation or cluster) matches
exactly the version used in the Singularity image. Otherwise, unwanted behavior like crashing or hanging
might be observed. Sometimes, a difference in the minor version is still OK, but I wouldn't rely on that.
The version installed on the image is **Open-MPI 4.0.3** (default package version for Ubuntu 20.04). To
check your default MPI version, run:

```
mpirun --version
```

The recommended way to run a simulation is as follows:

```
# make a run directory (ignored by version control)
mkdir -p run
# make a copy of the test case
cp -r test_cases/rhoCF_set1_alpha4_saiddes_ref1 run/
# perform the simulation with Singularity
cd run/rhoCF_set1_alpha4_saiddes_ref1
./Allrun
```

### Local OpenFOAM installation

A standard installation of [OpenFOAM-v2012](https://openfoam.com/download/) will be fine to simulate all of the test cases in this repository. No compilation of additional libraries oder applications is required. Executing a simulation follows the same steps as outlined above. Use the **Allrun.local** script instead of *Allrun*.

```
# make a run directory (ignored by version control)
mkdir -p run
# make a copy of the test case
cp -r test_cases/rhoCF_set1_alpha4_saiddes_ref1run/
# perform the simulation with Singularity
cd run/rhoCF_set1_alpha4_saiddes_ref1
./Allrun.local
```

### Test cases

Several simulation setups are provided under *test_cases*. Not all of them were used to produce the final data. However, they might be useful if further experimentation with the setup is desired. A few hints for the folder names:

- *rhoCF* indicates *rhoCentralFoam* as flow solver; all other cases use *rhoPimpleFoam*
- *set1* indicates the flow conditions described in the introduction
- *alpha* indicates the angle of attack
- *sa* stands for URANS simulations with Spalart-Allmaras closure; *saiddes* stands for hybrid IDDES modelling with Spalart-Allmaras closure
- *wf* indicates the usage of wall functions
- *zXX* indicates a 3D simulation with XX cells in spanwise direction
- *ref* indicates the refinement level

## References

First results were presented in the following conference article:
```
@inbook{doi:10.2514/6.2022-2591,
author = {Andre Weiner and Richard Semaan},
title = {Simulation and modal analysis of transonic shock buffets on a NACA-0012 airfoil},
booktitle = {AIAA SCITECH 2022 Forum},
chapter = {},
pages = {},
doi = {10.2514/6.2022-2591},
URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2022-2591},
eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2022-2591}
}
```

The library used for data processing and modal decomposition is called [flowTorch](https://github.com/FlowModelingControl/flowtorch):
```
@article{Weiner2021,
  doi = {10.21105/joss.03860},
  url = {https://doi.org/10.21105/joss.03860},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {68},
  pages = {3860},
  author = {Andre Weiner and Richard Semaan},
  title = {flowTorch - a Python library for analysis and reduced-order modeling of fluid flows},
  journal = {Journal of Open Source Software}
}
```
