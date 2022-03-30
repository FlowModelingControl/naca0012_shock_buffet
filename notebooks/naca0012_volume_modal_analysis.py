"""Compute POD of 3D simulations.
"""

from os.path import join
from flowtorch.data import HDF5Dataloader, HDF5Writer, copy_hdf5_mesh, mask_box, XDMFWriter
from flowtorch.analysis import SVD, DMD
import torch as pt

# data source
path = "/media/andre/Elements/naca0012_shock_buffet/run/rhoCF_set1_alpha4_saiddes_ref0_z50/"
source_file = "flowtorch.hdf5"

# create copy for post-processing
target_file_pod = "pod.hdf5"
target_file_dmd = "dmd.hdf5"
copy_hdf5_mesh(path, source_file, target_file_pod)
copy_hdf5_mesh(path, source_file, target_file_dmd)

# set up loader, mask, and weights
loader = HDF5Dataloader(join(path, source_file))
vertices = loader.vertices / 0.6010500
mask = mask_box(vertices, [-0.2, -0.3, -1], [3, 1, 1])
n_points = mask.sum().item()
every = 8
end = 690
n_times = len(loader.write_times[:end:every])
dm = pt.zeros(n_points*3, n_times)

# create data matrix
volsq = pt.masked_select(loader.weights, mask).sqrt()
for i, time in enumerate(loader.write_times[:end:every]):
    print(f"\rProcessing snapshot {i}", end="")
    loader = HDF5Dataloader(join(path, source_file))
    U = loader.load_snapshot("U", time)
    dm[:n_points, i] = pt.masked_select(U[:, 0], mask) * volsq
    dm[n_points:2*n_points, i] = pt.masked_select(U[:, 1], mask) * volsq
    dm[2*n_points:, i] = pt.masked_select(U[:, 2], mask) * volsq
    del loader

print("")
loader = HDF5Dataloader(join(path, source_file))

# perform POD
svd = SVD(dm, 20)
print(svd)

writer = HDF5Writer(join(path, target_file_pod))
writer.write("mask", mask.shape, mask.type(pt.int32), "1", pt.int32)

dummy = pt.zeros((mask.shape[0], 3))
for i in range(10):
    dummy[mask, 0] = svd.U[:n_points, i] / volsq
    dummy[mask, 1] = svd.U[n_points:2*n_points, i] / volsq
    dummy[mask, 2] = svd.U[2*n_points:, i] / volsq
    writer.write(f"mode_{i}", dummy.shape, dummy, "1", pt.float32)

# save also singular values and left singular vectors
outpath = "./output/naca0012_data/"
pt.save(svd.s, f"{outpath}svd_z50_sig.pt")
pt.save(svd.V, f"{outpath}svd_z50_V.pt")

del svd, writer


# perform DMD
dt = (float(loader.write_times[1]) - float(loader.write_times[0])) * every
dmd = DMD(dm, dt, optimal=True)
print(dmd)

writer = HDF5Writer(join(path, target_file_dmd))
writer.write("mask", mask.shape, mask.type(pt.int32), "1", pt.int32)
top_k = dmd.top_modes(50, integral=True)
top_k = [k for k in top_k if dmd.frequency[k] > 1.0]
print(dmd.frequency[pt.tensor(top_k[:9], dtype=pt.int64)])
for i, mi in enumerate(top_k):
    dummy[mask, 0] = dmd.modes[:n_points, mi].real / volsq
    dummy[mask, 1] = dmd.modes[n_points:2*n_points, mi].real / volsq
    dummy[mask, 2] = dmd.modes[2*n_points:, mi].real / volsq
    writer.write(f"mode_{i}_real", dummy.shape, dummy, "1", pt.float32)
    dummy[mask, 0] = dmd.modes[:n_points, mi].imag / volsq
    dummy[mask, 1] = dmd.modes[n_points:2*n_points, mi].imag / volsq
    dummy[mask, 2] = dmd.modes[2*n_points:, mi].imag / volsq
    writer.write(f"mode_{i}_imag", dummy.shape, dummy, "1", pt.float32)

# save also frequencies, importance, and eigenvalues
outpath = "./output/naca0012_data/"
pt.save(dmd.frequency, f"{outpath}dmd_z50_freq.pt")
pt.save(dmd.integral_contribution, f"{outpath}dmd_z50_int.pt")
pt.save(dmd.eigvals, f"{outpath}dmd_z50_eigvals.pt")

## save reconstruction of top 3 modes
#rec = dmd.partial_reconstruction(dmd.top_modes(3, integral=True))
#del dmd
#dummy = pt.zeros((mask.shape[0], 3))
#for i, t in enumerate(loader.write_times[:end:every]):
#    print(f"\rWriting reconstruction for time {t}", end="")
#    dummy[mask, 0] = rec[:n_points, i] / volsq
#    dummy[mask, 1] = rec[n_points:2*n_points, i] / volsq
#    dummy[mask, 2] = rec[2*n_points:, i] / volsq
#    writer.write(f"top3_U", dummy.shape, dummy, t, pt.float32)
#
#print()


# create XDMF wrapper
xdmf = XDMFWriter.from_filepath(join(path, target_file_pod))
xdmf.create_xdmf("pod.xdmf")
xdmf = XDMFWriter.from_filepath(join(path, target_file_dmd))
xdmf.create_xdmf("dmd.xdmf")
