"""Helper functions for data processing and visualization.
"""


import pandas as pd
import numpy as np
from glob import glob
from sklearn.neighbors import KNeighborsRegressor

TOL = 1.0e-6


def fetch_force_coefficients(path):
    """Load force coefficients.

    The following steps are executed:
    - find available time folders
    - load and merge coefficient data

    Parameters
    ----------
    path - str: path to location of time folders with
        force coefficients

    Returns
    -------
    t - array: time
    cd - array: drag coefficient
    cl - array: lift coefficient

    """
    if not path[-1] == "/":
        path = path + "/"
    times = glob(path + "*")
    times = sorted([t.split("/")[-1] for t in times], key=float)
    print("Found {:d} time folders in path {:s}".format(len(times), path))
    names = ["t", "Cd", "Cs", "Cl", "CmRoll", "CmPitch",
             "CmYaw", "Cd_f", "Cd_r", "Cs_f", "Cs_r", "Cl_f", "Cl_r"]
    df_list = []
    for tf in times:
        file_path = path + tf + "/coefficient.dat"
        df_list.append(pd.read_csv(file_path, sep="\t",
                                   skiprows=range(13), header=None, names=names))
    return np.split(pd.concat(df_list)[["t", "Cd", "Cl"]].values, 3, 1)


def fetch_surface_data(path, remove_te=False):
    """Load and process surface sample data.

    The following processing is done:
    - split into lower and upper side
    - normalization of x with chord length
    - sorting of all fields according to x_*

    Parameters
    ----------
    path - str: path to csv file
    remove_te - bool: trailing edge data are removed if True

    Returns
    -------
    x_* - array: normalized coordinate value along the airfoil for lower and upper side
    z_* - array: normalized coordinate value in spanwise direction for lower and upper side
    f_* - array: surface field for lower and upper side

    """
    data = pd.read_csv(path, sep=" ", skiprows=[
                       0, 1], header=None, names=["x", "y", "z", "f"])
    x_max = data.x.max()
    limit = 0.999 if remove_te else 1.1
    x_up = data[(data.y >= 0) & (data.x < limit*x_max)].x
    x_up = x_up.values / x_max
    z_up = data[(data.y >= 0) & (data.x < limit*x_max)].z
    z_up = z_up.values / x_max
    f_up = data[(data.y >= 0) & (data.x < limit*x_max)].f.values
    x_low = data[(data.y < 0) & (data.x < limit*x_max)].x
    x_low = x_low.values / x_max
    z_low = data[(data.y < 0) & (data.x < limit*x_max)].z
    z_low = z_low.values / x_max
    f_low = data[(data.y < 0) & (data.x < limit*x_max)].f.values
    ind_up = np.argsort(x_up)
    ind_low = np.argsort(x_low)
    return x_up[ind_up], z_up[ind_up], f_up[ind_up], x_low[ind_low], z_low[ind_low], f_low[ind_low]


def error_norms(x, f, x_ref, f_ref):
    """Compute L1, L2, and maximum norm for a 1D field.

    It is not necessary that x and x_ref have the same length.
    The error/deviation from the reference is computed as follows:
    - for a given x_ref-value find the two closed points in x,
      interpolate f, and compute the difference from f_ref
    - compute the error norms from the difference

    Parameters
    ----------
    x* - array: coordinate in airfoil length direction
    f* - array: field value to compare

    Returns
    -------
    l1 - float: L1 norm
    l2 - float: L2 norm
    lmax - float: maximum norm 

    """
    def linear_interpolation(x, x_1, x_2, f_1, f_2):
        dx_1 = x - x_1
        dx_2 = x - x_2
        return (f_1 * dx_2 + f_2 * dx_1) / (dx_1 + dx_2)

    f_int = np.zeros_like(f_ref)
    for i, x_i in enumerate(x_ref):
        idx = np.absolute(x_i - x).argmin()
        if abs(x[idx] - x_i) < TOL:
            f_int[i] = f[idx]
        elif x[idx] >= x_i:
            if idx > 0:
                f_int[i] = linear_interpolation(
                    x_i, x[idx-1], x[idx], f[idx-1], f[idx])
            else:
                f_int[i] = f[idx]
        else:
            if idx < x.shape[0] - 1:
                f_int[i] = linear_interpolation(
                    x_i, x[idx], x[idx+1], f[idx], f[idx+1])
            else:
                f_int[i] = f[idx]

    diff = np.absolute(f_int - f_ref)
    l1 = np.mean(diff)
    l2 = np.mean(np.square(diff))
    lmax = np.max(diff)
    return l1, l2, lmax


def spanwise_average(field, n_depth):
    """Compute spanwise average of surface field.

    Note: the function assumes that the mesh was created by
    extruding a 2D mesh linearly in spanwise direction with
    constant cell width.

    Parameters
    ----------
    field - array: surface field, sorted by ascending x
    n_depth - int: number of points in spanwise direction

    Returns
    -------
    field_av - array: surface field averaged in spanwise direction

    """
    assert field.shape[0] % n_depth == 0
    n_chord = int(field.shape[0] / n_depth)
    field_av = np.zeros(n_chord)
    for i in range(n_chord):
        field_av[i] = np.mean(field[i*n_depth:(i+1)*n_depth+1])
    return field_av


def spanwise_points(x):
    """Determine points in spanwise direction.

    Note: the function assumes that the mesh was created by
    extruding a 2D mesh in spanwise direction.

    Parameters
    ----------
    x - array: coordinate values in chordwise direction

    Returns
    -------
    n_z - int: number of points in spanwise direction

    """
    n_z = 1
    for i, x_i in enumerate(x):
        if abs(x[0] - x_i) > 1.0E-12:
            n_z = i
            break
    return n_z


def find_write_times(path):
    """Find all time folders in a given path

    Note: the location given by path must contains only time folders.

    Parameters:
    -----------
    path - str: location of time folders

    Returns
    -------
    times - list: sorted list of write times as string; the sorting is
        accending according to the numerical time value

    """
    if not path[-1] == "/":
        path = path + "/"
    times = glob(path + "*")
    times = sorted([t.split("/")[-1] for t in times], key=float)
    print("Found {:d} time folders in path {:s}".format(len(times), path))
    print("Available time range t={:s}...{:s}s".format(times[0], times[-1]))
    return times


def average_surface_data(path, file_name, t_start=0.0, t_end=1000.0, remove_te=False):
    """Average surface field in time and spanwise direction.

    The functions does the following processing steps:
    - find all available time folders
    - select all folders in the specified time window
    - load surface data and compute statistics

    Note: it is assumed that the mesh is a 2D mesh extruded
    in the third (spanwise, z) direction. Therefore, averaging
    in spanwise direction merges points with the same x-coordinate.

    Parameters
    ----------
    path - str: path to the location of all time folders
    file_name - str: name of file with surface data; must be
        the same for all time folders
    t_* float: start and end time for averaging window
    remove_te - bool: trailing edge data are removed if True

    Returns
    -------
    x_* - array: coordinate in airfoil length direction
    f_mean_* - array: time average for upper and lower side
    f_std_* - array: standard deviation for upper and lower side
    f_min_* - array: minimum value for upper and lower side
    f_max_* - array: maximum value for upper and lower side

    """
    times = find_write_times(path)
    i_start = min(range(len(times)), key=lambda i: abs(
        float(times[i]) - t_start))
    i_end = min(range(len(times)), key=lambda i: abs(float(times[i]) - t_end))
    print("Computing statistics for t={:s}...{:s}s ({:d} snapshots)".format(
        times[i_start], times[i_end], len(times[i_start:i_end])))
    x_up, _, f_up, x_low, _, f_low = fetch_surface_data(
        path + times[i_start] + "/" + file_name)
    n_z = spanwise_points(x_up)
    F_up = np.zeros((int(f_up.shape[0] / n_z), len(times[i_start:i_end])))
    F_low = np.zeros((int(f_low.shape[0] / n_z), len(times[i_start:i_end])))
    F_up[:, 0] = np.copy(spanwise_average(f_up, n_z))
    F_low[:, 0] = np.copy(spanwise_average(f_low, n_z))
    for i, t in enumerate(times[i_start+1:i_end]):
        _, _, f_up, _, _, f_low = fetch_surface_data(path + t + "/" + file_name, remove_te)
        F_up[:, i+1] = np.copy(spanwise_average(f_up, n_z))
        F_low[:, i+1] = np.copy(spanwise_average(f_low, n_z))
    return (spanwise_average(x_up, n_z), spanwise_average(x_low, n_z),
            np.mean(F_up, axis=1), np.mean(F_low, axis=1),
            np.std(F_up, axis=1), np.std(F_low, axis=1),
            np.min(F_up, axis=1), np.min(F_low, axis=1),
            np.max(F_up, axis=1), np.max(F_low, axis=1))


def interpolate_uniform_1D(times, samples, n_points, n_neighbors=2):
    """Create uniformly space samples in time by interpolation.

    Background: the time step in the numerical simulation changes
    over time. For frequency analysis of data that was sampled at
    each CFD time loop, the data must be interpolated to uniformly
    spaced time. The interpolation uses the K nearest neighbors
    algorithm (KNN).

    Parameters
    ----------
    times - array: sample times, typically not uniformfly spaced
    samples - array: samples taken at *times*
    n_points - int: number of uniformly spaced samples
    n_neighbors - int: number of neighbors to use for interpolation
        in the KNN algorithm

    Returns
    -------
    t - array: uniformly spaced times; length of n_points
    uniform_samples - array: interpolated samples; length of n_points

    """
    t = np.linspace(np.min(times), np.max(times), n_points)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(times, samples)
    uniform_samples = knn.predict(t[..., np.newaxis])
    return t, uniform_samples


def sample_surface_field(path, field_name, sample_x, sample_z, upper=True):
    """Sample a surface at a given location over time.

    The follow steps are executed to sample a surface field:
    - find all time folders
    - load first surface field and find point closest to sample location
    - load surface field for all remaining time folders and extract sample

    Parameters
    ----------
    path - str: location of surface field data
    field_name - str: name of field to sample
    sample_* - float: x or z component of sample location; must be given
        relative to chord length
    upper - bool: sample upper surface if True or lower surface if False

    Returns
    -------
    t - array: sample times as floats
    sample - array: surface field samples

    """
    times = find_write_times(path)
    assert len(times) > 0
    t = np.array([float(t_i) for t_i in times])
    sample = np.zeros_like(t)
    x_up, z_up, f_up, x_low, z_low, f_low = fetch_surface_data(path + times[0] + f"/{field_name}")
    if upper:
        dist = np.linalg.norm(np.vstack((x_up, z_up)).T - np.array([sample_x, sample_z]), axis=1)
        closest = np.argmin(dist)
        sample[0] = f_up[closest]
        field_ind = 2
        print("Sample location at x/c={:2.4f}, z/c={:2.4f} on upper side".format(
            np.round(x_up[closest], 4), np.round(z_up[closest], 4))
        )
    else:
        dist = np.linalg.norm(np.vstack((x_low, z_low)).T - np.array([sample_x, sample_z]), axis=1)
        closest = np.argmin(dist)
        sample[0] = f_low[closest]
        field_ind = 5
        print("Sample location at x/c={:2.4f}, z/c={:2.4f} on lower side".format(
            np.round(x_low[closest], 4), np.round(z_low[closest], 4))
        )

    for i, time in enumerate(times[1:]):
        data = fetch_surface_data(path + f"{time}/{field_name}")
        sample[i+1] = data[field_ind][closest]

    return t, sample


def compute_fft(t, probe, t_min, t_max, subtract_mean=True):
    """Compute the Fast Fourier Transform of a probed signal.

    A window in which to compute the FFT can be specified by
    providing start and end time. The following link is very
    helpful in understanding the NumPy convenience functions:
    https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python/27191172

    Parameters
    ----------
    t - array: sample times; assumed to be equally spaced (const. dt)
    probe - array: probe of which to compute the FFT
    t_min - float: start of the time window
    t_max - float: end of the time window
    subtract_mean - bool: subtract the probes mean in the window before
        computing the FFT; defaults to True

    Returns
    -------
    freq - array: frequency bin values
    amplitude - array: amplitudes corresponding to each frequency

    """
    # select values in time window
    assert t_min < t_max, "Start time must be smaller than end time."
    start_idx = np.absolute(t-t_min).argmin()
    end_idx = np.absolute(t-t_max).argmin()
    t_win = t[start_idx:end_idx+1]
    p_win = probe[start_idx:end_idx+1]
    # compute fft
    dt = t_win[1] - t_win[0]
    elapsed_time = t_win[-1] - t_win[0]
    n_samples = t_win.shape[0]
    df = 1.0 / elapsed_time  # frequency resolution
    f_max = df * n_samples / 2.0  # highest detectable frequency
    p_win = p_win - p_win.mean() if subtract_mean else p_win
    amplitude = np.fft.fft(p_win)
    freq = np.fft.fftfreq(amplitude.shape[0]) * n_samples * df
    print(f"Selected {n_samples} samples; frequency resolution df={df:2.4f}Hz; f_max={f_max:2.2f}Hz")
    pos_end = int(n_samples/2) if n_samples % 2 == 0 else int((n_samples+1)/2)
    return freq[:pos_end], np.absolute(amplitude)[:pos_end]


if __name__ == "__main__":
    pass
