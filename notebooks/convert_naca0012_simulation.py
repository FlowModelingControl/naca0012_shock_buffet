"""Convert 3D OpenFOAM simulation data to flowTorch HDF5 data.
"""

from flowtorch.data import FOAM2HDF5, FOAMCase

path = "/media/andre/Elements/naca0012_shock_buffet/run/rhoCF_set1_alpha4_saiddes_ref0_z50"
case = FOAMCase(path, False)
times = case._eval_write_times()
times = [t for t in times if 0.0326 <= float(t) <= 0.1015]
converter = FOAM2HDF5(path)
converter.convert("flowtorch.hdf5", ["U", "Ma", "rho"], times)