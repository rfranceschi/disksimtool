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

obs_path = Path('./observations/')
profiles_path = Path('./profiles/')

profiles_dict = {}
fpath = profiles_path / 'observed_profiles.h5'
with h5py.File(fpath, 'r') as f:
    for _key in f.keys():
        profiles_dict[_key] = {}
        for _profile_key in f[_key].keys():
            profiles_dict[_key][_profile_key] = f[_key][_profile_key][:]

@traced
def likelihood(params: list, **kwargs):
    model_dir = disk_model(params, model_options, show_plots=False)
    lh = images_likelihood(model_dir, **kwargs)
    # shutil.rmtree(model_dir)
    logging.info(lh)
    if lh is None:
        logging.warning('No likelihood was calculated, check if the number of'
                        'pixels is not too small to extract a radial profile.')
    return lh

@traced
def images_likelihood(model_path: Path, normalized_profiles: list = None,
                      r_norm_as: float = None, r_min: float = None) -> float:
    """
    Calculate the total chi-squared (chi2) value for a model's generated images
    against observed profiles.

    Parameters
    ----------
    model_path: Path
        Path to the folder containing the fits files.
    normalized_profiles: list, optional
        The name of the fits files whose profiles we need normalized.
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
    for output_fits in model_path.glob('*.fits'):
        obs_profile = profiles_dict[output_fits.stem].copy()

        y_obs = obs_profile['y']
        dy_obs = obs_profile['dy']

        rn_as = None
        if output_fits.stem in normalized_profiles:
            rn_as = r_norm_as
            rn_au = ((r_norm_as * u.arcsec).to_value(u.rad) * model_options[
                'distance_pc'] * u.pc).to_value(u.au)

            norm = np.interp(rn_au, obs_profile['x'], obs_profile['y'])
            y_obs /= norm
            dy_obs /= norm

        r_vals = obs_profile['x'] / (model_options['distance_pc'] *
                  u.pc).to_value(u.au)
        r_vals = (r_vals * u.rad).to_value(u.arcsec)

        i_outer = np.nonzero(np.asarray(r_vals > r_min))
        r_vals = r_vals[i_outer]
        x_obs = obs_profile['x'][i_outer]
        y_obs = y_obs[i_outer]
        dy_obs = dy_obs[i_outer]

        x_model, y_model, dy_model = model_utils.get_profile_from_fits(
            output_fits,
            clip=6,
            inc=model_options['inc'],
            PA=model_options['PA'],
            dist=model_options['distance_pc'],
            beam=None,
            r_norm=rn_as,
            r_min=r_min,
            rvals=r_vals,
        )

        chi2 += hf.calculate_chisquared(y_model,
                                        y_obs,
                                        dy_obs,
                                        )
        f, ax = plt.subplots()
        ax.semilogy((x_model * u.arcsec).to_value(u.rad)*(model_options[
            'distance_pc'] * u.pc).to_value(u.au),
                    y_model, '-',
                    color='k')
        ax.semilogy(x_obs, y_obs, '-', color='r')
        plt.suptitle(output_fits.stem)
        plt.show()
        # r_in_as = 0.5
        # condition = np.nonzero(np.asarray(x_model > r_in_as))
        # chi2 += hf.calculate_chisquared(y_model[condition],
        #                                 obs_profile['y'][condition],
        #                                 obs_profile['dy'][condition],
        #                                 )
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

    # grain size distribution exp
    lo = 0
    hi = 1
    # uniform prior
    params_transformed[0] = params[0] * (hi - lo) + lo

    # grain size distribution, a**(4-x)
    lo = 0
    hi = 20
    # uniform prior
    params_transformed[1] = params[1] * (hi - lo) + lo

    # grain size distribution, a**(4-x)
    lo = 0.001
    hi = 0.1
    # log prior
    params_transformed[2] = 10 ** (
                params[2] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))

    # d2g exp
    lo = 0
    hi = 20
    # uniform prior
    params_transformed[3] = params[1] * (hi - lo) + lo

    # d2g at 70 au
    lo = 0.001
    hi = 0.1
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
        'lam_obs_list': [0.0015],
        # Set scattering (True) or continuum (False) radiative transfer for
        'scattering': [False],
        'coord': '11h01m51.9053285064s -34d42m17.033218380s',
        'npix': 200,
        'threads': 16,
        'sigma_funct': sigma_funct,
    }

    model_params_names = ['size exp', 'amax exp', 'amax coeff', 'd2g exp',
                   'd2g coeff']
    model_params = [
        0.1,  # grain size distribution, the x in a**(4-x)
        4.,  # max grain size radial wdistribution exponent
        0.5,  # max grain size radial distribution coeff at options['r_c']
        7,  # d2g exp
        1.0,  # d2g at options['r_c']
    ]

    model_dir = disk_model(model_params, model_options, show_plots=False)
    shutil.rmtree(model_dir / 'radmc_run')
    os.remove(model_dir / 'model.pkl')
    print(model_dir)

    #  The 870 profile is normalized since there is likely an issue
    #  the units when extracting the profiles.
    # normalized_profiles = ['1.6_mu', '15.0_mu']
    # wrapped_likelihood = partial(likelihood,
    #                              # normalized_profiles=normalized_profiles,
    #                              r_norm_as=0.6,
    #                              r_min=0.1)
    #
    # print(wrapped_likelihood(model_params))

    # sampler = ultranest.ReactiveNestedSampler(param_names, wrapped_likelihood,
    #                                           prior_transform,
    #                                           log_dir="myanalysis")
    # results = sampler.run()
