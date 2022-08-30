"""Test sensitivity of DMD variants to the number of snapshots.
"""

from os import makedirs
import torch as pt
import pickle
from flowtorch.analysis import DMD
from utils import lhs_sampling_1d

# output of DMD results
output = "./output/naca0012_analysis/dmd_variants/"
makedirs(output, exist_ok=True)

# DMD variants
dmd_options = {
    "optDMD" : {"unitary" : False, "optimal" : True, "tlsq" : False},
    "optTDMD" : {"unitary" : False, "optimal" : True, "tlsq" : True},
    "optUDMD" : {"unitary" : True, "optimal" : True, "tlsq" : False}
}

every = (1, 2, 4, 8, 16, 32, 64)

# helper function to collect and save relevant data
def test_dmd_variant(dm, dt, options, key):
    eigs, freq, sort_int, sort_amp, err, top_k_err, top_k_err_int = [], [], [], [], [], [], []
    for e in every:
        print(f"\rUsing every {e} snapshot", end="")
        dmd = DMD(dm[:, ::e], dt*e, **options)
        eigs.append(dmd.eigvals)
        freq.append(dmd.frequency)
        sort_int.append(dmd.top_modes(100, integral=True))
        sort_amp.append(dmd.top_modes(100, integral=False))
        err.append(dmd.reconstruction_error.norm().item())
        top_k_err.append((dmd.partial_reconstruction(dmd.top_modes(21, integral=False)) - dm[:, ::e]).norm().item())
        top_k_err_int.append((dmd.partial_reconstruction(dmd.top_modes(21, integral=True)) - dm[:, ::e]).norm().item())
    to_save = [eigs, freq, sort_int, sort_amp, err, top_k_err, top_k_err_int]
    with open(f"{output}{key}.pkl", "wb") as out:
        pickle.dump(to_save, out, pickle.HIGHEST_PROTOCOL)
    print("")

# surface pressure
data = "./output/naca0012_data/"
dm_cp = pt.load(f"{data}surface/dm_ref1_z25.pt")
times_cp = pt.load(f"{data}surface/times_ref1_z25.pt")
start_idx = (times_cp - 0.025).abs().argmin()
end_idx = (times_cp - 0.1385).abs().argmin()
dm_cp = dm_cp[:, start_idx:end_idx+1]
dt_cp = (times_cp[1] - times_cp[0])

for key in dmd_options.keys():
    print(f"Testing {key} on surface pressure")
    test_dmd_variant(dm_cp, dt_cp, dmd_options[key], f"{key}_cp_snapshots")
    print("")
del dm_cp

# slice data
times = pt.load(f"{data}slice/times_ref1_z25.pt")
dt = (times[1] - times[0])
w = pt.load(f"{data}slice/w_ref1_z25.pt").sqrt().unsqueeze(-1)
n_points = w.shape[0]

##  density, slice, weighted, every second snapshot
dm_rho = pt.load(f"{data}slice/dm_ref1_z25.pt")[3*n_points:4*n_points, :] * w
for key in dmd_options.keys():
    print(f"Testing {key} on slice density")
    test_dmd_variant(dm_rho, dt, dmd_options[key], f"{key}_rho_snapshots")
    print("")
del dm_rho

##  x-y-velocity, slice, weighted, every second snapshot
dm_uxy = pt.load(f"{data}slice/dm_ref1_z25.pt")[:2*n_points, :] * w.repeat((2, 1))
for key in dmd_options.keys():
    print(f"Testing {key} on slice velocity")
    test_dmd_variant(dm_uxy, dt, dmd_options[key], f"{key}_uxy_snapshots")
    print("")
del dm_uxy

##  x-y-velocity, slice, weighted, every second snapshot
dm_uxya = pt.zeros((3*n_points, times.shape[0]))
dm_full = pt.load(f"{data}slice/dm_ref1_z25.pt")
dm_uxya[:2*n_points, :] = dm_full[:2*n_points, :] * w.repeat((2, 1))
kappa = pt.tensor(1.4)
scale = pt.sqrt(2.0 / (kappa * (kappa - 1.0)))
U = (dm_full[:n_points, :]**2 + dm_full[n_points:2*n_points, :]**2 + dm_full[2*n_points:3*n_points, :]**2).sqrt()
Ma = dm_full[4*n_points:, :]
dm_uxya[2*n_points:, :] = scale * (U / Ma) * w
del dm_full, U, Ma

for key in dmd_options.keys():
    print(f"Testing {key} on slice velocity/speed of sound")
    test_dmd_variant(dm_uxya, dt, dmd_options[key], f"{key}_uxya_snapshots")
    print("")
del dm_uxya