"""Extract a slice from 3D simulation and save as tensor.
"""

from os import makedirs
import torch as pt
from flowtorch.data import HDF5Dataloader, mask_box
from flow_conditions import CHORD


# create output folder
output = "./output/naca0012_data/slice/"
makedirs(output, exist_ok=True)

# raw data path and loader
path = "/media/andre/Elements/naca0012_shock_buffet/run/rhoCF_set1_alpha4_saiddes_ref1_z25/flowtorch.hdf5"
loader = HDF5Dataloader(path)

# times, points, mask, and volumes
vertices = loader.vertices / CHORD
zmin, zmax = -0.002 / CHORD, 0.002 / CHORD
mask = mask_box(vertices, [-0.2, -0.3, zmin], [3, 1, zmax])
n_points = mask.sum().item()
times = pt.tensor([float(t) for t in loader.write_times])
n_times = times.shape[0]

pt.save(pt.masked_select(vertices[:, 0], mask), f"{output}x_ref1_z25.pt")
pt.save(pt.masked_select(vertices[:, 1], mask), f"{output}y_ref1_z25.pt")
pt.save(pt.masked_select(loader.weights, mask), f"{output}w_ref1_z25.pt")
pt.save(times, f"{output}times_ref1_z25.pt")

# create data matrix
dm = pt.zeros((5*n_points, n_times))

for i, t in enumerate(loader.write_times):
    print(f"\rProcessing snapshot {i}, t={t}", end="")
    # re-creating and deleting the loader is a workaround for
    # a current bug in hdf5/h5py; see 
    # https://github.com/h5py/h5py/issues/2010
    # https://github.com/tenpy/hdf5_io/issues/2
    loader = HDF5Dataloader(path)
    U, Ma, rho = loader.load_snapshot(["U", "Ma", "rho"], t)
    dm[:n_points, i] = pt.masked_select(U[:, 0], mask)
    dm[n_points:2*n_points, i] = pt.masked_select(U[:, 1], mask)
    dm[2*n_points:3*n_points, i] = pt.masked_select(U[:, 2], mask)
    dm[3*n_points:4*n_points, i] = pt.masked_select(rho, mask)
    dm[4*n_points:, i] = pt.masked_select(Ma, mask)
    del loader

pt.save(dm, f"{output}dm_ref1_z25.pt")