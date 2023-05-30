"""Test rank sensitivity of DMD variants applied to different state vectors.
"""

from os import makedirs
from os.path import join
from collections import defaultdict
import torch as pt
from flowtorch.analysis import DMD
from utils import lhs_sampling_1d

# output of DMD results
output = "./output/oat15_analysis/dmd_variants/"
makedirs(output, exist_ok=True)

# DMD variants
dmd_options = {
    "DMD" : {"unitary" : False, "optimal" : False, "tlsq" : False},
    "optDMD" : {"unitary" : False, "optimal" : True, "tlsq" : False},
    "TDMD" : {"unitary" : False, "optimal" : False, "tlsq" : True},
    "optTDMD" : {"unitary" : False, "optimal" : True, "tlsq" : True},
    "UDMD" : {"unitary" : True, "optimal" : False, "tlsq" : False},
    "optUDMD" : {"unitary" : True, "optimal" : True, "tlsq" : False}
}

# ranks to test
ranks = lhs_sampling_1d(5, 200, 50)
pt.save(ranks, f"{output}ranks.pt")

# helper function to collect and save relevant data
def test_dmd_variant(dm, dt, options, key, n_points, weights):
    results = defaultdict(list)
    for r in ranks:
        print(f"\rTesting rank r={r}", end="")
        # unweighted data matrix
        dmd = DMD(dm, dt, rank=r, **options)
        results["eigvals"].append(dmd.eigvals)
        results["frequency"].append(dmd.frequency)
        results["importance"].append(dmd.integral_contribution)
        results["top_100_int"].append(dmd.top_modes(100, integral=True))
        results["top_100_amp"].append(dmd.top_modes(100, integral=False))
        # weighted data matrix
        dmd_w = DMD(dm*weights, dt, rank=r, **options)
        results["eigvals_w"].append(dmd_w.eigvals)
        results["frequency_w"].append(dmd_w.frequency)
        results["importance_w"].append(dmd_w.integral_contribution)
        results["top_100_int_w"].append(dmd_w.top_modes(100, integral=True))
        results["top_100_amp_w"].append(dmd_w.top_modes(100, integral=False))

        # reconstruction/prediction error with and without weighting
        ## unweighted error of unweighted data matrix
        err_uw_uw = dmd.reconstruction_error
        ## weighted error of unweighted data matrix
        err_w_uw = err_uw_uw * weights
        ## unweighted error of weighted data matrix
        err_uw_w= dmd_w.reconstruction / weights - dm
        ## weighted error of weighted data matrix
        err_w_w = dmd_w.reconstruction_error
        ## compute norms for individual components of state vector
        n_fields = int(dm.shape[0] / n_points)
        for fi in range(n_fields):
            # norm of mean state, weighted and unweighted
            mean_norm = dm[fi*n_points:(fi+1)*n_points, :].mean(dim=1).norm()
            mean_norm_w = (dm*weights)[fi*n_points:(fi+1)*n_points, :].mean(dim=1).norm()
            # error norms
            ## error unweighted, data matrix unweighted
            err_uw_uw_fi = err_uw_uw[fi*n_points:(fi+1)*n_points, :].norm()
            results[f"rec_err_uw_uw_{fi}"].append(err_uw_uw_fi.item())
            results[f"rec_err_uw_uw_norm_{fi}"].append((err_uw_uw_fi/mean_norm).item())
            ## error weighted, data matrix unweighted
            err_w_uw_fi = err_w_uw[fi*n_points:(fi+1)*n_points, :].norm()
            results[f"rec_err_w_uw_{fi}"].append(err_w_uw_fi.item())
            results[f"rec_err_w_uw_norm_{fi}"].append((err_w_uw_fi/mean_norm_w).item())
            ## error unweighted, data matrix weighted
            err_uw_w_fi = err_uw_w[fi*n_points:(fi+1)*n_points, :].norm()
            results[f"rec_err_uw_w_{fi}"].append(err_uw_w_fi.item())
            results[f"rec_err_uw_w_norm_{fi}"].append((err_uw_w_fi/mean_norm).item())
            ## error weighted, data matrix weighted
            err_w_w_fi = err_w_w[fi*n_points:(fi+1)*n_points, :].norm()
            results[f"rec_err_w_w_{fi}"].append(err_w_w_fi.item())
            results[f"rec_err_w_w_norm_{fi}"].append((err_w_w_fi/mean_norm_w).item())

        # projection error with and without weighting
        YH = (dmd.modes @ pt.diag(dmd.eigvals)) @ \
            (pt.linalg.pinv(dmd.modes) @ dmd._dm[:, :-1].type(dmd.modes.dtype))
        YH_w = (dmd_w.modes @ pt.diag(dmd_w.eigvals)) @ \
            (pt.linalg.pinv(dmd_w.modes) @ dmd_w._dm[:, :-1].type(dmd_w.modes.dtype))
        ## error unweighted, data matrix unweighted
        err_uw_uw = dm[:, 1:] - YH.real.type(dmd._dm.dtype)
        ## error weighted, data matrix unweighted
        err_w_uw = err_uw_uw * weights
        ## error unweighted, data matrix weighted
        err_uw_w = dm[:, 1:] - YH_w.real.type(dmd_w._dm.dtype) / weights
        ## error weighted, data matrix weighted
        err_w_w = dm[:, 1:] * weights - YH_w.real.type(dmd_w._dm.dtype)
        ## compute norms for individual components of state vector
        for fi in range(n_fields):
            # norm of mean state, weighted and unweighted
            mean_norm = dm[fi*n_points:(fi+1)*n_points, 1:].mean(dim=1).norm()
            mean_norm_w = (dm * weights)[fi*n_points:(fi+1)*n_points, 1:].mean(dim=1).norm()
            # error norms
            ## error unweighted, data matrix unweighted
            err_uw_uw_fi = err_uw_uw[fi*n_points:(fi+1)*n_points, :].norm()
            results[f"pro_err_uw_uw_{fi}"].append(err_uw_uw_fi.item())
            results[f"pro_err_uw_uw_norm_{fi}"].append((err_uw_uw_fi/mean_norm).item())
            ## error weighted, data matrix unweighted
            err_w_uw_fi = err_w_uw[fi*n_points:(fi+1)*n_points, :].norm()
            results[f"pro_err_w_uw_{fi}"].append(err_w_uw_fi.item())
            results[f"pro_err_w_uw_norm_{fi}"].append((err_w_uw_fi/mean_norm_w).item())
            ## error unweighted, data matrix weighted
            err_uw_w_fi = err_uw_w[fi*n_points:(fi+1)*n_points, :].norm()
            results[f"pro_err_uw_w_{fi}"].append(err_uw_w_fi.item())
            results[f"pro_err_uw_w_norm_{fi}"].append((err_uw_w_fi/mean_norm).item())
            ## error weighted, data matrix weighted
            err_w_w_fi = err_w_w[fi*n_points:(fi+1)*n_points, :].norm()
            results[f"pro_err_w_w_{fi}"].append(err_w_w_fi.item())
            results[f"pro_err_w_w_norm_{fi}"].append((err_w_w_fi/mean_norm_w).item())

    pt.save(results, join(output, f"{key}.pt",))
    print("")
    
# slice data, common
data = "/home/andre/Development/naca0012_shock_buffet/run/oat15"
times = pt.load(join(data, "oat15_tandem_times.pt"))[::20]
dt = times[1] - times[0]
vertices = pt.load(join(data, "vertices_and_masks.pt"))
area = vertices["area_small"]
del vertices
start_at, end_at = 101, 501 # encloses 2 cycles
n_points = area.shape[0]

# density
dm = pt.load(join(data, "rho_small_every10.pt"))[:, start_at:end_at:2]
weights = area.sqrt().unsqueeze(-1)
for key in dmd_options.keys():
    print(f"Testing {key} on density")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_rho", n_points, weights)
    print("")


# velocity in x and z
vel_x = pt.load(join(data, "vel_x_small_every10.pt"))[:, start_at:end_at:2]
vel_z = pt.load(join(data, "vel_z_small_every10.pt"))[:, start_at:end_at:2]
dm = pt.cat((vel_x, vel_z), dim=0)
del vel_x, vel_z
weights = area.sqrt().unsqueeze(-1).repeat((2, 1))
for key in dmd_options.keys():
    print(f"Testing {key} on vel. x and z")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_vel_xz", n_points, weights)
    print("")


# velocity in x, y, and z
vel_x = pt.load(join(data, "vel_x_small_every10.pt"))[:, start_at:end_at:2]
vel_y = pt.load(join(data, "vel_y_small_every10.pt"))[:, start_at:end_at:2]
vel_z = pt.load(join(data, "vel_z_small_every10.pt"))[:, start_at:end_at:2]
dm = pt.cat((vel_x, vel_y, vel_z), dim=0)
del vel_x, vel_y, vel_z
weights = area.sqrt().unsqueeze(-1).repeat((3, 1))
for key in dmd_options.keys():
    print(f"Testing {key} on vel. x, y and z")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_vel_xyz", n_points, weights)
    print("")


# velocity in x and z, local speed of sound
vel_x = pt.load(join(data, "vel_x_small_every10.pt"))[:, start_at:end_at:2]
vel_y = pt.load(join(data, "vel_y_small_every10.pt"))[:, start_at:end_at:2]
vel_z = pt.load(join(data, "vel_z_small_every10.pt"))[:, start_at:end_at:2]
ma = pt.load(join(data, "ma_small_every10.pt"))[:, start_at:end_at:2]
speed = (vel_x**2 + vel_y**2 + vel_z**2).sqrt()
a_loc = speed / ma
kappa = pt.tensor(1.4)
scale = pt.sqrt(2.0 / (kappa * (kappa - 1.0)))
dm = pt.cat((vel_x, vel_z, a_loc*scale), dim=0)
del speed, a_loc, vel_x, vel_y, vel_z, ma
weights = area.sqrt().unsqueeze(-1).repeat((3, 1))
for key in dmd_options.keys():
    print(f"Testing {key} on vel. x, z and a")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_vel_axz", n_points, weights)
    print("")