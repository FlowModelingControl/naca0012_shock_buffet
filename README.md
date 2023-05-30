
![FOR2895Logo](for2895_logo.png)

# A robust dynamic mode decomposition methodology for an airfoil undergoing transonic shock buffet

This repository accompanies the following publications:

- [1] **A. Weiner, R. Semaan:** [A robust dynamic mode decomposition methodology for an airfoil undergoing transonic shock buffet](https://arxiv.org/abs/2212.10250), accepted for publication in the *AIAA Journal* (2023)
- [2] **A. Weiner, R. Semaan:** [Simulation and modal analysis of transonic shock buffets on a NACA-0012 airfoil](https://arc.aiaa.org/doi/abs/10.2514/6.2022-2591), *AIAA SCITECH* (2022)

The simulation setups and analysis tools available in this repository were created as part of the research program [FOR 2895 - Unsteady flow interaction phenomena at high speed stall conditions](https://www.for2895.uni-stuttgart.de/) financed by the German Research Foundation (DFG). The primary intention of this repository is to provide a fully reproducible workflow for the simulation and modal analysis of transonic shock buffets.

## flowTorch analysis package

The main library for data processing and modal decomposition is [flowTorch](https://github.com/FlowModelingControl/flowtorch). FlowTorch and additional dependencies can be installed via *pip*:

```
pip3 install git+https://github.com/FlowModelingControl/flowtorch@aweiner
```

## OAT15A analysis

The final analysis reported in [1] was conducted on simulation data of an OAT15A airfoil. The simulations were performed by a project partner; refer to [this report](https://arxiv.org/abs/2301.05760). This repository contains the following scripts and notebooks (note that not all notebook cells are executed due to the proprietary OAT15A geometry):

- [oat15_state_vectors.ipynb](notebooks/oat15_state_vectors.ipynb): comparison of weighted and unweighted state vector norms
- [oat15_local_error.ipynb](notebooks/oat15_local_error.ipynb): local projection error (in space) with and without volume/area weighting
- [oat15_dmd_variants.ipynb](notebooks/oat15_dmd_variants.ipynb): rank sensitivity of various state vectors and DMD variants evaluated in terms of prediction/projection error, dominant eigenvalues/frequencies, and full DMD spectra; requires the execution of [test_dmd_rank_sensitivity_oat15.py](notebooks/test_dmd_rank_sensitivity_oat15.py) first
- [oat15_sample_frequency_analysis.ipynb](notebooks/oat15_sample_frequency_analysis.ipynb): sampling rate dependency of full DMD spectrum
- [oat15_dmd_analysis.ipynb](notebooks/oat15_dmd_analysis.ipynb): DMD analysis of various state vectors (density, pressure, velocity, local speed of sound); propagation speeds of vortex shedding and buffet modes; rank dependency of unitary DMD modes
- [oat15_greedy_selection.ipynb](notebooks/oat15_greedy_selection.ipynb): comparison of greedy mode selection and integral criterion

Due to the proprietary nature of the OAT15A airfoil, we can not make the full dataset available for public download. If you are interested in the datasets, please get in touch via [mail](mailto:andre.weiner@yahoo.de) or [LinkedIn](https://www.linkedin.com/in/andre-weiner-a79752133/).

## NACA0012 analysis and simulation setups

### Analysis scripts and notebooks

An [earlier version](https://arxiv.org/abs/2212.10250v1) of [1] and the analysis reported in [2] were based on NACA0012 simulation data and the following scripts/notebooks. If the notebooks are not displayed correctly in the browser, clone the repository and open the notebooks locally.

- [naca0012_dmd_rank_spectra.ipynb](notebooks/naca0012_dmd_rank_spectra.ipynb): rank sensitivity of full DMD spectrum for various state vectors and DMD variants
- [naca0012_dmd_spectogram.ipynb](notebooks/naca0012_dmd_spectogram.ipynb): influence of the number of buffet cycles in combination with small changes in the data and rank truncation on the full DMD spectrum
- [naca0012_dmd_variants.ipynb](notebooks/naca0012_dmd_variants.ipynb): analysis of rank sensitivity of various state vectors and DMD variants in terms of prediction error and dominant eigenvalues/frequencies; requires running *test_dmd_\*.py* scripts first
- [naca0012_animate_dmd_modes.ipynb](notebooks/naca0012_animate_dmd_modes.ipynb): partial reconstruction and animation of selected DMD modes
- [naca0012_lift_and_drag.ipynb](notebooks/naca0012_lift_and_drag.ipynb): visual inspection of lift and drag coefficients of 2D and 3D simulations; frequency analysis
- [naca0012_pressure_coefficients.ipynb](notebooks/naca0012_pressure_coefficients.ipynb): analysis of surface pressure coefficients; mesh dependency study
- [naca0012_probe_analysis.ipynb](notebooks/naca0012_probe_analysis.ipynb): frequency analysis of flow speed at selected probe locations
- [naca0012_schlieren_analysis.ipynb](notebooks/naca0012_schlieren_analysis.ipynb): spatio-temporal correlation coefficients of a line sample extracted from numerical Schlieren images
- [naca0012_slice_modal_analysis.ipynb](notebooks/naca0012_slice_modal_analysis.ipynb): modal analysis of slice data; spatio-temporal correlation coefficients of line sample
- [naca0012_state_vectors.ipynb](notebooks/naca0012_state_vectors.ipynb): norms of various state vectors
- [naca0012_surface_modal_analysis.ipynb](notebooks/naca0012_surface_modal_analysis.ipynb): modal analysis of surface pressure coefficients
- [naca0012_volume_modal_analysis.ipynb](notebooks/naca0012_volume_modal_analysis.ipynb): visualization of the modal decomposition computed with *naca0012_volume_modal_analysis.py*
- *extract_lift_and_drag.py*: combines, cleans, and stores the force coefficient data of an OpenFOAM simulation
- *utils.py*: helper functions for loading, processing, analysis, and plotting
- *convert_naca0012_simulations.py*: converts a reconstructed OpenFOAM simulation into flowTorch HDF5 format
- *create_naca0012_slice_dm.py*: extracts a slice from an OpenFOAM simulation and stores the snapshots in a data matrix
- *create_naca0012_surface_dm.py*: extracts the pressure coefficient on the airfoil's upper surface and stores the snapshot data in data matrices

### NACA0012 simulations

All simulations can be executed using Singularity or a local installation of OpenFOAM-v2012. Instructions and dependencies for each workflow follow below.

#### Local OpenFOAM installation

A standard installation of [OpenFOAM-v2012](https://openfoam.com/download/) will be fine to simulate all of the test cases in this repository. No compilation of additional libraries oder applications is required. Use the **Allrun.local** script instead of *Allrun* if you are not using the Singularity container:

```
# make a run directory (ignored by version control)
mkdir -p run
# make a copy of the test case
cp -r test_cases/rhoCF_set1_alpha4_saiddes_ref1 run/
# perform the simulation
cd run/rhoCF_set1_alpha4_saiddes_ref1
./Allrun.local
```

#### Singularity

[Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) is a container tool that allows making results reproducible and performing simulations, to a large extent, platform independent. The only remaining dependencies are Singularity itself and OpenMPI (see next section for further comments). To build the image, run:

```
sudo singularity build of_v2012.sif docker://andreweiner/of_pytorch:of2012-py1.7.1-cpu
```

The resulting image file *of_v2012.sif* should be located at the repository's top level.

When executing an application in parallel using the Singularity image, it is important that the host's OpenMPI version (installed on your workstation or cluster) matches exactly the version used in the Singularity image. Otherwise, unwanted behavior like crashing or hanging might be observed. Sometimes, a difference in the minor version is still OK, but I wouldn't rely on that. The version installed on the image is **Open-MPI 4.0.3** (default package version for Ubuntu 20.04). To check your default MPI version, run:

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

#### Test cases

Several simulation setups are provided under *test_cases*. Not all of them were used to produce the final data. However, they might be useful if further experimentation with the setup is desired. A few hints regarding the folder names:

- *rhoCF* indicates *rhoCentralFoam* as flow solver; all other cases use *rhoPimpleFoam*
- *set1* corresponds to the flow conditions reported in [2]
- *alpha* indicates the angle of attack
- *sa* stands for URANS simulations with Spalart-Allmaras closure; *saiddes* stands for hybrid IDDES modelling with Spalart-Allmaras closure
- *wf* indicates the usage of wall functions
- *zXX* indicates a 3D simulation with XX cells in spanwise direction
- *ref* indicates the refinement level
