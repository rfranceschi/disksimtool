from pathlib import Path
import pickle

from astropy import constants as c

from menu_model import disk_model


M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
au = c.au.cgs.value

obs_path = Path('./observations/')
profile_path = Path('./profiles/')

model_options = {
        'mstar': 0.8 * M_sun,
        'lstar': 1 * L_sun,
        'tstar': 3810,
        'nr': 1000,
        'rin': 0.32 * au,
        'rout': 61.7 * au,
        'alpha': 1e-5,
        'fname_opac': '/Users/rfranceschi/mysims/LESIA/opacities/dustkappa_p30_chopped.npz',
        'inc': 7,
        'PA': 0,
        'distance': 56,
        # The output fits files will be at these wavelengths
        'lam_obs_list': [0.000165, 0.0015, 0.087],
        # Scattered light (True) or continuum (False) emission for lam_obs_list
        'scattering': [True, False, False],
        'coord': '11h01m51.9053285064s -34d42m17.033218380s'
    }

profiles_dict = {}
for _profile in profile_path.iterdir():
    with open(_profile, 'rb') as f:
        profiles_dict[_profile.stem] = pickle.load(f)


def likelihood(params: list):
    model_path = disk_model(params, model_options)

