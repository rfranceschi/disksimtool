import logging
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import ultranest
from astropy import constants as c

from menu_model import disk_model
from disksimtool import helper_functions as hf
from disksimtool import model_utils

M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
au = c.au.cgs.value

obs_path = Path('./observations/')
profile_path = Path('./profiles/')

model_options = {
    'mstar': 0.8 * M_sun,
    'lstar': 1 * L_sun,
    'tstar': 3810,
    'nr': 400,
    'rin': 0.32 * au,
    'rout': 250 * au,
    'alpha': 1e-5,
    'fname_opac': 'opacities/dustkappa_p30_chopped.npz',
    'inc': 7,
    'PA': 0,
    'distance': 56,
    # The output fits files will be at these wavelengths (micron)
    'lam_obs_list': [0.000165, 0.0015, 0.087],
    # Set scattering (True) or continuum (False) radiative transfer for
    # lam_obs_list wavelengths
    'scattering': [True, False, False],
    'coord': '11h01m51.9053285064s -34d42m17.033218380s',
    'npix': 500,
    'threads': 16,
}

profiles_dict = {}
for _profile in profile_path.iterdir():
    with open(_profile, 'rb') as f:
        profile_key = _profile.stem.split('_', 1)[1]
        profiles_dict[profile_key] = pickle.load(f)


def likelihood(params: list):
    model_dir = disk_model(params, model_options)
    lh = images_likelihood(model_dir)
    shutil.rmtree(model_dir)
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
    for _profile_key in profiles_dict.keys():
        model_fits = model_path / ('image_' + _profile_key + '.fits')
        if not model_fits.exists():
            warnings.warn(f"No model fits found at wavelength {_profile_key}")
            return -np.inf
        profile = profiles_dict[_profile_key]
        x_model, y_model, dy_model = model_utils.get_profile_from_fits(
            model_path,
            clip=6,
            inc=model_options['inc'],
            PA=model_options['PA'],
            beam=profile['beam'],
            )
        chi2 += hf.calculate_chisquared(y_model, profile['y'], profile['yerr'])
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
    param_names = ['size exp', 'amax exp', 'amax coeff', 'd2g exp',
                   'd2g coeff']
    sampler = ultranest.ReactiveNestedSampler(param_names, likelihood,
                                              prior_transform,
                                              log_dir="myanalysis")
    results = sampler.run()
