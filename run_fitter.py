import logging
import pickle
import shutil
import warnings
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import ultranest
from astropy import constants as c
from astropy import units as u
from matplotlib import pyplot as plt

from menu_model import disk_model, sigma_with_rim
from disksimtool import helper_functions as hf
from disksimtool import model_utils

M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
au = c.au.cgs.value

obs_path = Path('./observations/')
profiles_path = Path('./profiles/')

model_options = {
    'mstar': 0.8 * M_sun,
    'lstar': 1 * L_sun,
    'tstar': 3810,
    'nr': 400,
    'rin': 0.32 * au,
    'rout': 250 * au,
    'r_c': 50 * au,
    'alpha': 1e-5,
    'fname_opac': 'opacities/dustkappa_p30_chopped.npz',
    'inc': 7,
    'PA': 0,
    'distance_pc': 56,
    # The output fits files will be at these wavelengths (micron)
    'lam_obs_list': [0.000165, 0.087],
    # Set scattering (True) or continuum (False) radiative transfer for
    # lam_obs_list wavelengths
    'scattering': [True, False],
    'coord': '11h01m51.9053285064s -34d42m17.033218380s',
    'npix': 500,
    'threads': 16,
}

profiles_dict = {}
fpath = profiles_path / 'observed_profiles.h5'
with h5py.File(fpath, 'r') as f:
    for _key in f.keys():
        profiles_dict[_key] = {}
        for _profile_key in f[_key].keys():
            profiles_dict[_key][_profile_key] = f[_key][_profile_key][:]


def likelihood(params: list):
    model_dir = disk_model(params, model_options)
    lh = images_likelihood(model_dir)
    # shutil.rmtree(model_dir)
    logging.info(lh)
    return lh


def images_likelihood(model_path: Path) -> float:
    """
    Returns the total chi2 for the model images corresponding to the
    observed profiles.

    Parameters
    ----------
    model_path: Path

    Returns
    -------
    float

    """
    chi2 = 0


    for output_fits in model_path.glob('*.fits'):
        obs_profile = profiles_dict[output_fits.stem]

        x_model, y_model, dy_model = model_utils.get_profile_from_fits(
            output_fits,
            clip=6,
            inc=model_options['inc'],
            PA=model_options['PA'],
            dist=model_options['distance_pc'],
            beam=obs_profile['beam'],
        )

        r_in_as = 0.5
        condition = np.nonzero(np.asarray(x_model > r_in_as))
        chi2 += hf.calculate_chisquared(y_model[condition],
                                        obs_profile['y'][condition],
                                        obs_profile['dy'][condition])
        return chi2

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


if __name__ == '__main__':
    params = {
        'sigma_exp': 24,
        'r_exp': 3.1 * au,
        'p': 0.5,
        'w': 0.45,
    }
    sigma_funct = partial(sigma_with_rim, **params)
    model_options['sigma_funct'] = sigma_funct

    param_names = ['size exp', 'amax exp', 'amax coeff', 'd2g exp',
                   'd2g coeff']
    sampler = ultranest.ReactiveNestedSampler(param_names, likelihood,
                                              prior_transform,
                                              log_dir="myanalysis")
    results = sampler.run()
