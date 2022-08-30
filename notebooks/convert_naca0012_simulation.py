"""Convert 3D OpenFOAM simulation data to flowTorch HDF5 data.
"""

from flowtorch.data import FOAM2HDF5, FOAMCase

path = "/media/andre/Elements/naca0012_shock_buffet/run/rhoCF_set1_alpha4_saiddes_ref1_z25"
case = FOAMCase(path, False)
times = case._eval_write_times()
times = [t for t in times if 0.025 <= float(t) <= 0.1385]
converter = FOAM2HDF5(path)
converter.convert("flowtorch.hdf5", ["U", "Ma", "rho"], times)