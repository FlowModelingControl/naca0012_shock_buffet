"""Extract a slice from 3D simulation and save as tensor.
"""

from os import makedirs
import torch as pt
from flowtorch.data import FOAMDataloader, mask_box
from flow_conditions import CHORD


# create output folder
output = "./output/naca0012_data/slice/"
makedirs(output, exist_ok=True)

# raw data path and loader
path = "/media/andre/Elements/naca0012_shock_buffet/run/rhoCF_set1_alpha4_saiddes_ref1_z25/"
loader = FOAMDataloader(path, distributed=False)

# times, points, mask, and volumes
vertices = loader.vertices / CHORD
zmin, zmax = -0.002 / CHORD, 0.002 / CHORD
mask = mask_box(vertices, [-0.2, -0.3, zmin], [3, 1, zmax])
n_points = mask.sum().item()
n_times = len(loader.write_times)

# create data matrix, load and mask, and save
dm = pt.zeros((n_points, n_times))

for i, t in enumerate(loader.write_times):
    print(f"\rProcessing snapshot {i}, t={t}", end="")
    dm[:, i] = pt.masked_select(loader.load_snapshot("grad(rho)", t).norm(dim=1), mask)

pt.save(dm, f"{output}magGradRho_ref1_z25.pt")