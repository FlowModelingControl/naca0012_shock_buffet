"""Test rank sensitivity of DMD variants applied to different state vectors.
"""

from os import makedirs
from os.path import join
from collections import defaultdict
import torch as pt
from flowtorch.analysis import DMD
from utils import lhs_sampling_1d

# output of DMD results
output = "./output/naca0012_analysis/dmd_variants/"
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
ranks = lhs_sampling_1d(5, 500, 100)
pt.save(ranks, f"{output}ranks_ref1_z25.pt")

# helper function to collect and save relevant data
def test_dmd_variant(dm, dt, options, key, n_points, weights):
    results = defaultdict(list)
    for r in ranks:
        print(f"\rTesting rank r={r}", end="")
        dmd = DMD(dm*weights, dt, rank=r, **options)
        results["eigvals"].append(dmd.eigvals)
        results["frequency"].append(dmd.frequency)
        results["importance"].append(dmd.integral_contribution)
        results["top_100_int"].append(dmd.top_modes(100, integral=True))
        results["top_100_amp"].append(dmd.top_modes(100, integral=False))
        rec_err = dmd.reconstruction / weights - dm
        n_fields = int(dm.shape[0] / n_points)
        for fi in range(n_fields):
            mean_norm = dm[fi*n_points:(fi+1)*n_points, :].mean(dim=1).norm()
            err_fi = rec_err[fi*n_points:(fi+1)*n_points, :].norm()
            results[f"rec_err_{fi}"].append(err_fi.item())
            results[f"rec_err_norm_{fi}"].append((err_fi/mean_norm).item())
        results["rec_err"].append(dmd.reconstruction_error.norm().item())
        YH = (dmd.modes @ pt.diag(dmd.eigvals)) @ \
            (pt.linalg.pinv(dmd.modes) @ dmd._dm[:, :-1].type(dmd.modes.dtype))
        p_err = (dmd._dm[:, 1:] - YH.real.type(dmd._dm.dtype)).norm().item()
        results["pro_err"].append(p_err)
    pt.save(results, join(output, f"{key}.pt",))
    print("")
    
# surface pressure
data = "./output/naca0012_data/"
dm_cp = pt.load(f"{data}surface/dm_ref1_z25.pt")
times_cp = pt.load(f"{data}surface/times_ref1_z25.pt")
start_idx = (times_cp - 0.025).abs().argmin()
end_idx = (times_cp - 0.1385).abs().argmin()
# use only every second snapshot; yields about 110 snapshots per cycle
dm_cp = dm_cp[:, start_idx:end_idx+1:2]
dt_cp = (times_cp[1] - times_cp[0]) * 2

for key in dmd_options.keys():
    print(f"Testing {key} on surface pressure, unweighted")
    test_dmd_variant(dm_cp, dt_cp, dmd_options[key], f"{key}_cp")
    print("")
del dm_cp

# slice data
times = pt.load(f"{data}slice/times_ref1_z25.pt")
dt = (times[1] - times[0]) * 2
w = pt.load(f"{data}slice/w_ref1_z25.pt").sqrt().unsqueeze(-1)
n_points = w.shape[0]

##  density, slice, unweighted
dm = pt.load(f"{data}slice/dm_ref1_z25.pt")[3*n_points:4*n_points, ::2]
for key in dmd_options.keys():
    print(f"Testing {key} on density, unweighted")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_rho")
    print("")

##  density, slice, weighted
dm *= w
for key in dmd_options.keys():
    print(f"Testing {key} on density, weighted")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_rho_weighted")
    print("")

##  x-y-velocity, slice, unweighted
dm = pt.load(f"{data}slice/dm_ref1_z25.pt")[:2*n_points, ::2]
for key in dmd_options.keys():
    print(f"Testing {key} on slice x-y velocity, unweighted")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_vel_xy")
    print("")

##  x-y-velocity, slice, unweighted
dm *= w.repeat((2, 1))
for key in dmd_options.keys():
    print(f"Testing {key} on slice x-y velocity, weighted")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_vel_xy_weighted")
    print("")

##  x-y-z-velocity, slice, unweighted
dm = pt.load(f"{data}slice/dm_ref1_z25.pt")[:3*n_points, ::2]
for key in dmd_options.keys():
    print(f"Testing {key} on slice x-y-z velocity, unweighted")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_vel_xyz")
    print("")

##  x-y-z-velocity, slice, weighted
dm *= w.repeat((3, 1))
for key in dmd_options.keys():
    print(f"Testing {key} on slice x-y-z velocity, weighted")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_vel_xyz_weighted")
    print("")

##  x-y-velocity + local speed of sound, slice, unweighted
dm = pt.zeros((3*n_points, times[::2].shape[0]))
dm_full = pt.load(f"{data}slice/dm_ref1_z25.pt")[:, ::2]
dm[:2*n_points, :] = dm_full[:2*n_points, :]
kappa = pt.tensor(1.4)
scale = pt.sqrt(2.0 / (kappa * (kappa - 1.0)))
U = (dm_full[:n_points, :]**2 + dm_full[n_points:2*n_points, :]**2 + dm_full[2*n_points:3*n_points, :]**2).sqrt()
Ma = dm_full[4*n_points:, :]
dm[2*n_points:, :] = scale * (U / Ma)
del dm_full, U, Ma

for key in dmd_options.keys():
    print(f"Testing {key} on velocity/speed of sound, unweighted")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_vel_axy")
    print("")

##  x-y-velocity + local speed of sound, slice, weighted
dm *= w.repeat((3, 1))
for key in dmd_options.keys():
    print(f"Testing {key} on velocity/speed of sound, weighted")
    test_dmd_variant(dm, dt, dmd_options[key], f"{key}_vel_axy_weighted")
    print("")
