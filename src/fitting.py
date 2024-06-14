import numpy as np


def calculate_chisquared(sim_data: np.array, obs_data: np.array, error: np.array):
    """

    Args:
        sim_data:
        obs_data:
        error:

    Returns:

    """

    error = error + 1e-100
    return np.sum((obs_data - sim_data) ** 2 / (error ** 2))
