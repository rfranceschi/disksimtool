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
def calculate_log_likelihood(sim_data: np.array, obs_data: np.array,
                         error: np.array) -> float:
    """
        Computes normalized chi-squared log-likelihood:
            ln L = -0.5 * sum((obs - sim)^2 / error^2) / N

        Args:
            sim_data: simulated data array
            obs_data: observed data array
            error: observational uncertainty for each point (same shape)

        Returns:
            Normalized log-likelihood
        """
    # Avoid zero division or NaNs
    error = np.where(error == 0, 1e-10, error)
    valid = np.isfinite(obs_data) & np.isfinite(sim_data) & np.isfinite(error)

    # Subselect only valid (non-NaN, finite) values
    resid2 = ((obs_data[valid] - sim_data[valid]) / error[valid]) ** 2

    # Return normalized log-likelihood
    return -0.5 * np.sum(resid2) / len(resid2) / 1e16
