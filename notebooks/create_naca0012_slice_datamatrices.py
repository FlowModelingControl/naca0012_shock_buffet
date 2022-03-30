"""Extract a slice from 3D simulation and save various data matrices.
"""

from os import makedirs
import torch as pt
from flowtorch.data import HDF5Dataloader, mask_box

# create output folder
output = "./output/naca0012_data/"
makedirs(output, exist_ok=True)

# raw data
path = "/media/andre/Elements/naca0012_shock_buffet/run/rhoCF_set1_alpha4_saiddes_ref0_z25/flowtorch.hdf5"
loader = HDF5Dataloader(path)
n_times = len(loader.write_times)

# points, mask, and volumes
chord = 0.6010500
vertices = loader.vertices / chord
zmin, zmax = 0.0024042/chord, 0.0072126/chord
mask = mask_box(vertices, [-0.2, -0.3, zmin], [3, 1, zmax])
n_points = mask.sum().item()
vol = loader.weights
volsq = pt.masked_select(vol, mask).sqrt()

# create data matrices
## density
rdm = pt.zeros((n_points, n_times))
## x and y velocity components
vdm = pt.zeros((2*n_points, n_times))
## velocity and speed of sound
avdm = pt.zeros((3*n_points, n_times))

for i, t in enumerate(loader.write_times):
    print(f"\rProcessing snapshot {i}", end="")
    # re-creating and deleting the loader is a workaround for
    # a current bug in hdf5/h5py; see 
    # https://github.com/h5py/h5py/issues/2010
    # https://github.com/tenpy/hdf5_io/issues/2
    loader = HDF5Dataloader(path)
    U, Ma, rho = loader.load_snapshot(["U", "Ma", "rho"], t)
    Ux = pt.masked_select(U[:, 0], mask)
    Uy = pt.masked_select(U[:, 1], mask)
    rhom = pt.masked_select(rho, mask)
    Um = (Ux**2 + Uy**2).sqrt()
    a = Um / pt.masked_select(Ma, mask)
    rdm [:, i] = rhom[:]
    vdm[:n_points, i] = Ux * volsq
    vdm[n_points:, i] = Uy * volsq
    avdm[:n_points, i] = Ux * volsq
    avdm[n_points:2*n_points, i] = Uy * volsq
    avdm[2*n_points:, i] = 2.0 / 0.4 * a * volsq
    del loader

pt.save(rdm, f"{output}rdm_slice.pt")
pt.save(vdm, f"{output}vdm_slice.pt")
pt.save(avdm, f"{output}avdm_slice.pt")