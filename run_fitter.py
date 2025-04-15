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
import ultranest

from menu_model import disk_model, sigma_with_rim
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

@traced
def likelihood(params: list, **kwargs) -> float:
    try:
        logging.info(f"Compute likelihood at params={params}")
        model_dir = disk_model(params, model_options, show_plots=False)
        lh = images_likelihood(model_dir, **kwargs)
        if lh is None:
            logging.warning('No likelihood was calculated at params={params}')
            lh = -1e300
    except Exception as e:
        logging.warning(f"Error at params={params}: {e}")
        lh = -1e300
    shutil.rmtree(model_dir, ignore_errors=True)
    return lh

@traced
def images_likelihood(model_path: Path, normalized_profiles: list = None,
                      r_norm_as: float = None, r_min: float = None,
                      plot: bool = False) -> float:
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

    chi2 = 0
    if plot:
        plt.close('all')
        f, ax = plt.subplots(2, 1)
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

        partial_chi2 = hf.calculate_chisquared(y_model,
                                        y_obs,
                                        dy_obs,
                                        )
        chi2 += partial_chi2

        if plot:
            ax[i].semilogy(x_model,
                        y_model, '-',
                        color='k',
                        label=f"{partial_chi2:.2e}")
            ax[i].semilogy(x_obs,
                        y_obs,
                        '-',
                        color='r')
            ax[i].set_title(output_fits.stem)
            ax[i].legend(fontsize='small')
        # r_in_as = 0.5
        # condition = np.nonzero(np.asarray(x_model > r_in_as))
        # chi2 += hf.calculate_chisquared(y_model[condition],
        #                                 obs_profile['y'][condition],
        #                                 obs_profile['dy'][condition],
        #                                 )
    if plot:
        title = model_path.name
        title = title.removeprefix("model_")
        f.text(0.3, 0.6, f'{chi2:.2e}', size='small')
        f.text(0.3, 0.95, title, size='small')
        plt.show()
    return chi2

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

    # grain size distribution exp, as in a**(4-exp)
    lo = 0
    hi = 1
    # uniform prior
    params_transformed[0] = params[0] * (hi - lo) + lo

    # grain size distribution exp, as in a0 * (r / r0)**exp
    lo = 2.5
    hi = 7.5
    # uniform prior
    params_transformed[1] = params[1] * (hi - lo) + lo

    # grain size distribution a0, as in a0 * (r / r0)**exp
    lo = 0.01
    hi = 1
    # log prior
    params_transformed[2] = 10 ** (
                params[2] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))

    # d2g exp
    lo = 0
    hi = 10
    # uniform prior
    params_transformed[3] = params[1] * (hi - lo) + lo

    # d2g at 70 au
    lo = 0.01
    hi = 1
    # log prior
    params_transformed[4] = 10 ** (
                params[2] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))

    return params_transformed

def run_and_clean(params):
    model_dir = disk_model(params, model_options, show_plots=False)
    shutil.rmtree(str(model_dir / 'radmc_run'))
    os.remove(str(model_dir / 'model.pkl'))


if __name__ == '__main__':
    params = {
        'sigma_exp': 24,
        'r_exp': 3.1 * au,
        'p': 0.5,
        'w': 0.45,
    }
    sigma_funct = partial(sigma_with_rim, **params)
    model_options = {
        'mstar': 0.75 * M_sun,
        'lstar': 0.242 * L_sun,
        'tstar': 3810,
        'nr': 250,
        'rin': 0.32 * au,
        'rout': 250 * au,
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
        'npix': 200,
        'threads': 1,
        'sigma_funct': sigma_funct,
    }

    model_params_names = [
        'size exp',
        'amax exp',
        'amax coeff',
        'd2g exp',
        'd2g coeff',
    ]

    normalized_profiles = ['1.6_mu']
    wrapped_likelihood = partial(likelihood,
                                 normalized_profiles=normalized_profiles,
                                 r_norm_as=0.6,
                                 r_min=0.4,
                                 plot=False)
    # likelihood = wrapped_likelihood(model_params)

    # normalized_profiles = ['1.6_mu', '15.0_mu']
    # wrapped_likelihood = partial(likelihood,
    #                              # normalized_profiles=normalized_profiles,
    #                              r_norm_as=0.6,
    #                              r_min=0.1)
    #
    # print(wrapped_likelihood(model_params))

    if "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ or "MPI_LOCALNRANKID" in os.environ:
        # Likely running under MPI
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        print(f"MPI rank {rank} started")

    sampler = ultranest.ReactiveNestedSampler(model_params_names,
                                              wrapped_likelihood,
                                              prior_transform,
                                              log_dir="myanalysis",
                                              # vectorized=True,
                                              resume=True,
                                              )
    results = sampler.run(Lepsilon=0.5,
                          min_num_live_points=1000,
                          dlogz=1.0,
                          show_status=True,
                          log_interval=1,
                          )

    try:
        sampler.print_results()
        ultranest.plot.cornerplot(results)
        plt.savefig('corner.png')
    except:
        warnings.warn('Something went wrong.')
