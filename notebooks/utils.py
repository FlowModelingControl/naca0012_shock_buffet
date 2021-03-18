"""Helper functions for data processing and visualization.
"""


import pandas as pd


def fetch_surface_data(path):
    """Load and process surface sample data.
    
    The following processing is done:
    - split into lower and upper side
    - normalization of x with chord length
    - removel of trailing edge data
    - removel up duplicate data in z-direction
    
    Parameters
    ----------
    path - str: path to csv file
    
    Returns
    -------
    x_* - array: normalized coordinate value along the airfoil for lower and upper side
    field_* - array: surface field for lower and upper side
    
    """
    data = pd.read_csv(path, sep=" ", skiprows=[0, 1], header=None, names=["x", "y", "z", "f"])
    x_max = data.x.max()
    x_up = data[ (data.y >= 0) & (data.z > 0) & (data.x < 0.999*x_max)].x
    x_up = x_up.values / x_max
    f_up = data[ (data.y >= 0) & (data.z > 0) & (data.x < 0.999*x_max) ].f.values
    x_low = data[ (data.y < 0) & (data.z > 0) & (data.x < 0.999*x_max) ].x
    x_low = x_low.values / x_max
    f_low = data[ (data.y < 0) & (data.z > 0) & (data.x < 0.999*x_max) ].f.values
    return x_up, f_up, x_low, f_low

if __name__ == "__main__":
    pass
