import multiprocessing
import os
from functools import partial
import logging
from pathlib import Path
import pickle
import shutil
import sys
import warnings

from astropy import constants as c
from astropy import units as u
from autologging import TRACE, traced
import gofish as gf
import h5py
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import ultranest

from menu_model import disk_model
from disksimtool import helper_functions as hf
from disksimtool import model_utils


logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stdout,
    format='%(levelname)s:%(name)s:%(funcName)s:%(message)s'
)

M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
au = c.au.cgs.value

distance = 56 * u.pc
incl = 7
PA = 0

profiles_path = Path('./profiles/')

profiles_dict = {}
fpath = profiles_path / 'observed_profiles.h5'
with h5py.File(fpath, 'r') as f:
    for _key in f.keys():
        profiles_dict[_key] = {}
        for _profile_key in f[_key].keys():
            profiles_dict[_key][_profile_key] = f[_key][_profile_key][:]

def log_prior_diag(theta: ArrayLike, mu: ArrayLike, sigma: ArrayLike):
    """
    Diagonal (independent) Gaussian prior.
    - theta, mu, sigma are 1D arrays of same length.
    - sigma must be > 0 for all elements.
    Returns the log prior (up to an additive constant).
    """
    theta = np.array(theta)
    mu = np.array(mu)
    sigma = np.array(sigma)
    if theta.shape != mu.shape or mu.shape != sigma.shape:
        raise ValueError("theta, mu and sigma must have the same shape")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    diff = (theta - mu) / sigma
    return -0.5 * np.sum(diff * diff)


@traced
def log_likelihood(params: list, **kwargs) -> float:
    try:
        logging.info(f"Compute likelihood at params={params}")
        model_dir = disk_model(params, model_options, show_plots=False)
        logL = images_log_likelihood(model_dir, params, **kwargs)
        if logL is None:
            logging.warning('No likelihood was calculated at params={params}')
            logL = -1e300
    except Exception as e:
        logging.warning(f"Error at params={params}: {e}")
        logging.warning(e)
        logL = -1e300
    shutil.rmtree(model_dir, ignore_errors=True)
    mu = [3.05, 6.250400805278292, 0.4495865608589366,
         2.714376852103456, 0.012578747337948015]
    sigma = [0.3, 1, 0.1, 0.5, 0.003]

    return logL + log_prior_diag(params, mu, sigma)

@traced
def images_log_likelihood(model_path: Path,
                      params: list,
                      normalized_profiles: list = None,
                      r_norm_as: float = None, r_min: float = None,
                      ) -> float:
    """
    Calculate the total chi-squared (chi2) value for a model's generated images
    against observed profiles.

    Parameters
    ----------
    model_path: Path
        Folder containing the fits files.
    normalized_profiles: list, optional
        Name of the fits files whose profiles we need normalized.
    r_norm_as: float, optional
        The radius, in arcsec, at which the profiles are normalized.
    r_min: float, optional

    Returns
    -------
    float

    """
    if not (normalized_profiles is None) == (r_norm_as is None):
        raise ValueError('Provide both or neither r_norm_as and normalized_profiles.')

    logL = 0
    # with h5py.File(output_file, 'a') as f:
    # Generate a unique name for this step (UUID or increment counter)
    # step_id = str(len(f))
    # step_group = f.create_group(step_id)
    # step_group.create_dataset('params', data=params)
    for i, output_fits in enumerate(model_path.glob('*.fits')):
        obs_profile = profiles_dict[output_fits.stem].copy()

        x_obs = np.copy(obs_profile['x'])
        y_obs = np.copy(obs_profile['y'])
        dy_obs = np.copy(obs_profile['dy'])

        r_norm = None
        if output_fits.stem in normalized_profiles:
            r_norm = r_norm_as
            norm = np.interp(r_norm_as, obs_profile['x'], obs_profile['y'])
            y_obs /= norm
            dy_obs /= norm

        i_inner = np.nonzero(np.asarray(x_obs > r_min))
        x_obs = x_obs[i_inner]
        y_obs = y_obs[i_inner]
        dy_obs = dy_obs[i_inner]

        r_max = 1.5
        i_outer = np.nonzero(np.asarray(x_obs < r_max))
        x_obs = x_obs[i_outer]
        y_obs = y_obs[i_outer]
        dy_obs = dy_obs[i_outer]

        x_model, y_model, dy_model, norm = model_utils.get_profile_from_fits(
            output_fits,
            inc=model_options['inc'],
            PA=model_options['PA'],
            dist=model_options['distance_pc'],
            beam=obs_profile['beam'],
            r_norm=r_norm,
            r_min=r_min,
            rvals=x_obs,
        )

        # step_group.create_dataset(f'{output_fits.stem}_x', data=x_model)
        # step_group.create_dataset(f'{output_fits.stem}_y', data=y_model)

        partial_logL = hf.calculate_log_likelihood(y_model, y_obs, dy_obs)
        logL += partial_logL

    return logL

@traced
def prior_transform(params: list) -> np.array:
    """
    Normalize the parameters prior to a [0,1] range
    Parameters
    ----------
    params

    Returns
    -------

    """
    params_transformed = np.copy(params)

    # grain size distribution exp, as in a**(-exp)
    lo = 2
    hi = 8
    # uniform prior
    params_transformed[0] = params[0] * (hi - lo) + lo

    # grain size distribution exp, as in a0 * (r / r0)**exp
    lo = 0
    hi = 10
    # uniform prior
    params_transformed[1] = params[1] * (hi - lo) + lo

    # grain size distribution a0, as in a0 * (r / r0)**exp
    lo = 0.0001
    hi = 0.1
    # log prior
    params_transformed[2] = 10 ** (
                params[2] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))

    # d2g exp
    lo = 0
    hi = 8
    # uniform prior
    params_transformed[3] = params[3] * (hi - lo) + lo

    # d2g at 70 au
    lo = 0.0001
    hi = 0.1
    # log prior
    params_transformed[4] = 10 ** (
                params[4] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))

    return params_transformed

def run_and_clean(params):
    model_dir = disk_model(params, model_options, show_plots=False)
    shutil.rmtree(str(model_dir / 'radmc_run'))
    os.remove(str(model_dir / 'model.pkl'))


if __name__ == '__main__':
    # params = {
    #     'sigma_exp': 24,
    #     'r_exp': 3.1 * au,
    #     'p': 0.5,
    #     'w': 0.45,
    # }
    # sigma_funct = partial(sigma_with_rim, **params)

    model_options = {
        'mstar': 0.75 * M_sun,
        'lstar': 0.242 * L_sun,
        'tstar': 3810,
        'nr': 250,
        'rin': 0.32 * au,
        'rout': 100 * au,
        'r_c': 30 * au,
        'alpha': 1e-3,
        'fname_opac': 'opacities/dustkappa_p30_chopped.npz',
        'inc': 7,
        'PA': 0,
        'distance_pc': 56,
        # The output fits files will be at these wavelengths (micron)
        # lam_obs_list wavelengths
        # 'lam_obs_list': [0.000165, 0.0015, 0.087],
        'lam_obs_list': [0.000165, 0.087],
        # Set scattering (True) or continuum (False) radiative transfer for
        'scattering': [True, False],
        'coord': '11h01m51.9053285064s -34d42m17.033218380s',
        'npix': 59,
        'threads': 1,

    }

    params_sigma = {
        'sigma_coeff': 118,
        'r_c': 45 * au,
        'r_exp': 3.1 * au,
        'gamma': 0.5,
        'w': 0.5,
    }
    sigma_funct = partial(model_utils.lbp_profile_with_rim, **params_sigma)
    model_options['sigma_funct'] = sigma_funct

    model_params_names = [
        'size exp',
        'amax exp',
        'amax coeff',
        'd2g exp',
        'd2g coeff',
    ]

    normalized_profiles = ['1.6_mu']
    wrapped_likelihood = partial(log_likelihood,
                                 normalized_profiles=normalized_profiles,
                                 r_norm_as=0.6,
                                 r_min=0.4,
                                 )

    if "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ or "MPI_LOCALNRANKID" in os.environ:
        # Likely running under MPI
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        print(f"MPI rank {rank} started")

    output_dir = Path("myanalysis")
    output_file = output_dir / "output.h5"
    sampler = ultranest.ReactiveNestedSampler(model_params_names,
                                              wrapped_likelihood,
                                              prior_transform,
                                              log_dir=str(output_dir),
                                              resume='resume',
                                              )
    results = sampler.run(Lepsilon=0.1,
                          min_num_live_points=800,
                          dlogz=0.1,
                          show_status=True,
                          log_interval=1,
                          )

    try:
        sampler.print_results()
        ultranest.plot.cornerplot(results)
        plt.savefig('corner.png')
    except:
        warnings.warn('Something went wrong.')
