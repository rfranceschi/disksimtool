import astropy.constants as c
import numpy as np
from autologging import traced

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value

def running_average(a, n=1):
    b = np.concatenate((np.ones(n) * a[0], a, np.ones(n) * a[-1]))
    return np.convolve(b, np.ones(2 * n + 1) / (2 * n + 1), mode='valid')

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

@traced
def calculate_chisquared(sim_data: np.array, obs_data: np.array,
                         error: np.array):
    """

    Args:
        sim_data:
        obs_data:
        error:

    Returns:

    """

    error = error + 1e-100
    obs_data = np.nan_to_num(obs_data, nan=0.0)
    sim_data = np.nan_to_num(sim_data, nan=0.0)
    chi2 = np.sum((obs_data - sim_data) ** 2 / (error ** 2))
    chi2 /= len(sim_data)
    return chi2
