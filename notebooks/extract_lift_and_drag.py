"""Load and pre-process lift and drag coefficients.

The simulations were re-started many times, and each run created
a different file containing the aerodynamic coefficients. Moreover,
the coefficient data was written at every numerical time step, but
the time step is not necessarily constant.

The script assembles the coefficient files, interpolates to much fewer
uniform time steps, and save the data into PyTorch tensors for convenient
post-processing.

"""

from os import makedirs
from os.path import join
from glob import glob
import torch as pt
from utils import fetch_force_coefficients, interpolate_uniform_1D


# data paths
raw_data_base = "/media/andre/Elements/naca0012_shock_buffet/run/"
output_path_2D = "./output/naca0012_data/2D/"
output_path_3D = "./output/naca0012_data/3D/"

# create output directories
makedirs(output_path_2D, exist_ok=True)
makedirs(output_path_3D, exist_ok=True)

# process relevant 2D simulations
cases_2D = ["rhoCF_set1_alpha4_saiddes_ref{:1d}".format(i) for i in (0, 1, 2)]
every = 1000
for i, case in enumerate(cases_2D):
    path = join(raw_data_base, case, "postProcessing/forces/")
    t, cd, cl = fetch_force_coefficients(path)
    ti, cl = interpolate_uniform_1D(t[::every], cl[::every], 10000)
    _, cd = interpolate_uniform_1D(t[::every], cd[::every], 10000)
    pt.save(pt.from_numpy(ti), join(output_path_2D, f"t_int_ref{i:1d}.pt"))
    pt.save(pt.from_numpy(cl), join(output_path_2D, f"cl_ref{i:1d}.pt"))
    pt.save(pt.from_numpy(cd), join(output_path_2D, f"cd_ref{i:1d}.pt"))

# process relevant 3D simulations
suffixes = ["ref0_z25", "ref0_z50", "ref1_z25"]
cases_3D = ["rhoCF_set1_alpha4_saiddes_{:s}".format(s) for s in suffixes]
every = 1000
for i, (case, name) in enumerate(zip(cases_3D, suffixes)):
    path = join(raw_data_base, case, "postProcessing/forces/")
    t, cd, cl = fetch_force_coefficients(path)
    # forgot to adjust area in z50 simulation (doubled spanwise width)
    fac = 0.5 if "z50" in case else 1.0
    ti, cl = interpolate_uniform_1D(t[::every], cl[::every]*fac, 10000)
    _, cd = interpolate_uniform_1D(t[::every], cd[::every]*fac, 10000)
    pt.save(pt.from_numpy(ti), join(output_path_3D, f"t_int_{name}.pt"))
    pt.save(pt.from_numpy(cl), join(output_path_3D, f"cl_{name}.pt"))
    pt.save(pt.from_numpy(cd), join(output_path_3D, f"cd_{name}.pt"))

