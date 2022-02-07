# OpenFOAM simulations of transonic shock buffets

The simulation setups and analysis tools available in this repository were created as part of the research program [FOR 2895](https://www.for2895.uni-stuttgart.de/) *Unsteady flow interaction phenomena at high speed stall conditions* financed by the German Research Foundation (DFG). The primary intention of this repository is to provide a fully reproducible workflow for the simulation and analysis of transonic shock buffets.

Note that the repository is still under construction. The workflow will be refined further and new data will be added in regular intervals.

All simulations can be executed using Singularity or a local installation of OpenFOAM-v2012. Instructions and dependencies for each workflow follow below.

## Singularity (recommended)

### Getting the OpenFOAM Singularity image

[Singularity]() is a container tool that allows making results reproducible and performing simulations, to a large extent, platform independent. The only remaining dependencies are Singularity itself and Open-MPI (see next section for further comments). To build the image, run:

```
sudo singularity build of_v2012.sif docker://andreweiner/of_pytorch:of2012-py1.7.1-cpu
```

The resulting image file *of_v2012.sif* should be located at the repository's top level.

### Running a simulation

When executing *rhoPimpleFoam* or any other application in parallel using the Singularity image,
it is important that the host's open-MPI version (installed on your workstation or cluster) matches
exactly the version used in the Singularity image. Otherwise, unwanted behavior like crashing or hanging
might be observed. Sometimes, a difference in the minor version is still OK, but I wouldn't rely on that.
The version installed in the image is **Open-MPI 4.0.3** (default package version for Ubuntu 20.04). To
check your default MPI version, run:

```
mpirun --version
```

The recommended way to run a simulation is as follows:

```
# make a run directory (ignored by version control)
mkdir -p run
# make a copy of the test case
cp -r test_cases/set1_alpha2_iddes_spalding_g5000 run/
# perform the simulation with Singularity
cd run/set1_alpha2_iddes_spalding_g5000
./Allrun
```

## Local OpenFOAM installation

A standard installation of [OpenFOAM-v2012](https://openfoam.com/download/) should be fine to simulate all of the test cases in this repository. No compilation of additional libraries oder applications is required. Executing a simulation follows the same steps as outlined above. Use the **Allrun.local** script instead of *Allrun*.

```
# make a run directory (ignored by version control)
mkdir -p run
# make a copy of the test case
cp -r test_cases/set1_alpha2_iddes_spalding_g5000 run/
# perform the simulation with Singularity
cd run/set1_alpha2_iddes_spalding_g5000
./Allrun.local
```

## References

```
@inbook{doi:10.2514/6.2022-2591,
author = {Andre Weiner and Richard Semaan},
title = {Simulation and modal analysis of transonic shock buffets on a NACA-0012 airfoil},
booktitle = {AIAA SCITECH 2022 Forum},
chapter = {},
pages = {},
doi = {10.2514/6.2022-2591},
URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2022-2591},
eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2022-2591},
    abstract = { View Video Presentation: https://doi.org/10.2514/6.2022-2591.vidWe perform 2D and 3D simulations of a NACA-0012 airfoil at pre- and post-buffet onset employing an IDDES simulation approach with Spalart-Allmaras closure model. The 3D simulations provide more realistic results in terms of buffet frequency and shock motion, highlighting the need for three-dimensional and scale-resolving simulations to properly capture the flow physics of the transonic buffet phenomenon. Dynamic mode decomposition identifies flow structures in the shock, boundary layer, and wake region that are slaved by the buffet cycle. We provide a fully reproducible, fully open-source simulation workflow to investigate transonic shock buffets. The workflow includes tools for visualization and modal analysis. }
}
```
