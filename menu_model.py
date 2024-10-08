import logging
import pickle
import warnings
from functools import partial
from pathlib import Path

import astropy.constants as c
import disklab.radmc3d
import dsharp_opac as do
import numpy as np
from radmc3dPy import image
from scipy.integrate import simpson

import disksimtool.model_utils as model_utils
import disksimtool.opac as opac
import disksimtool.radmc_utils as radmc_utils

logging.basicConfig(level=logging.INFO)
radmc3d_exec = Path('~/bin/radmc3d').expanduser()

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value


# distance = 56 * u.pc
# incl = 0
# PA = 0


def sigma_with_rim(r: float, sigma_exp: float, r_exp: float, p: float,
                   w: float) -> float:
    """
    Computes the surface density with an inner rim, as in Eq.(4) in Menu et
    al. 2014 (https://arxiv.org/pdf/1402.6597).

    Parameters
    ----------
    r: float
        Radial position.
    sigma_exp: float
        Normalization coefficient.
    r_exp: float
        Radial position where the outer disk starts.
    p: float
        Exponent of the outer disk profile.
    w: float
        Dimensionless rim width.

    Returns
    -------
    float
        Surface density.
    """
    r_dimensionless = r / r_exp
    outer_disk_density = sigma_exp * r_dimensionless ** -p

    inner_rim_mask = r_dimensionless < 1
    surface_density = outer_disk_density * np.ones_like(r)
    surface_density[inner_rim_mask] *= np.exp(
        -((1 - r_dimensionless[inner_rim_mask]) / w) ** 3)

    return surface_density


def integrate_sigma(r: np.array, sigma: np.array) -> float:
    """
    Compute the total mass by integrating a surface density profile.

    Parameters
    ----------
    r: np.array
    sigma: np.array

    Returns
    -------

    """
    return simpson(y=2 * np.pi * r * sigma, x=r)


def disk_model(parameters: list, options: dict) -> Path:
    """

    Parameters
    ----------
    parameters: list
        Free parameters.
    options: list
        Fixed parameters.

    Returns
    -------
    Path to the model directory.
    """
    # OPACITIES
    # Read if they exist, or calculate
    try:
        opac_dict = opac.read_opacs(
            Path(options['fname_opac']))
        # Double check that the wavelengths at which we want to compute the
        # images are in the opacity lambda array.
        lam_opac = opac_dict['lam']
        n_a = len(opac_dict['a'])
        for i, _lam in enumerate(options['lam_obs_list']):
            ilam_array = np.where(opac_dict['lam'] == _lam)[0]
            if ilam_array.size == 0:
                logging.warning(
                    "The observation lambda is not in the opacity lambda "
                    "array.")
    except FileNotFoundError:
        logging.warning("Opacity file not found, calculating opacities.")
        # Define the wavelength, size, and angle grids then calculate
        # opacities_IMLup and store them in a local file,
        # if it doesn't exist yet. Careful, that takes of the order of >2h.
        n_lam = 200  # number of wavelength points
        n_a = 30  # number of particle sizes
        n_theta = 181  # number of angles in the scattering phase function
        porosity = 0.3

        # wavelength and particle sizes grids
        lam_opac = np.logspace(-5, 0, n_lam)
        # We insert the observation wavelengths to be sure we don't need
        # interpolation
        for _lam_obs in options['lam_obs_list']:
            ilam = np.abs(lam_opac - _lam_obs).argmin()
            lam_opac[ilam] = _lam_obs

        opac.compute_opac(lam_opac, n_a, n_theta, porosity)
        opac_dict = opac.read_opacs(
            Path('opacities/dustkappa_p30_chopped.npz'))

    # DISK MODEL
    # Profile from Menu et al. 2014 (https://arxiv.org/pdf/1402.6597).
    r = np.linspace(options['rin'], options['rout'], 1000)
    density_params = {
        'sigma_exp': 24,
        'r_exp': 3.1 * au,
        'p': 0.5,
        'w': 0.45,
    }
    profile = sigma_with_rim(r, **density_params)
    disk_gas_mass = (integrate_sigma(r, profile) / c.M_sun.cgs.value)
    logging.info(f'Total disk mass: {disk_gas_mass:.2} M_sun')

    models_root = Path('./runs/')

    model_name = 'model_' + '_'.join([f'{_par:.5f}' for _par in parameters])
    model_path = models_root / model_name / 'model.pkl'

    if model_path.is_file():
        with open(model_path, 'rb') as fff:
            logging.warning(f'Loading model from {model_path}.')
            disk2d = pickle.load(fff)
    else:
        logging.info(f'Writing to {model_name} directory.')
        # Surface density parameters fixed from Menu et al. 2014 (
        # https://arxiv.org/pdf/1402.6597).
        density_func = partial(sigma_with_rim, **density_params)

        disk2d = model_utils.make_disklab2d_model(
            parameters,
            options['mstar'],
            options['lstar'],
            options['tstar'],
            options['nr'],
            options['alpha'],
            options['rin'],
            options['rout'],
            options['r_c'],
            options['fname_opac'],
            density_func,
            show_plots=False
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as fff:
            pickle.dump(disk2d, fff)

    # IMAGE RADIATIVE TRANSFER
    radmcfolder = model_path.parents[0] / 'radmc_run/'
    radmcfolder.mkdir(parents=True, exist_ok=True)
    radmc_utils.write_radmc3d(disk2d, lam_opac, radmcfolder, show_plots=False)

    # write the detailed scattering matrix files
    for i_grain in range(n_a):
        do.write_radmc3d_scatmat_file(i_grain, opac_dict, f'{i_grain}',
                                      path=radmcfolder)

    with open(Path(radmcfolder) / 'dustopac.inp', 'w') as f:
        disklab.radmc3d.write(f, '2               Format number of this file')
        disklab.radmc3d.write(f,
                              '{}              Nr of dust species'.format(n_a))

        for i_grain in range(n_a):
            disklab.radmc3d.write(f,
                                  '============================================================================')
            disklab.radmc3d.write(f,
                                  '10               Way in which this dust '
                                  'species is read')
            disklab.radmc3d.write(f, '0               0=Thermal grain')
            disklab.radmc3d.write(f,
                                  '{}              Extension of name of '
                                  'dustscatmat_***.inp file'.format(
                                      i_grain))

        disklab.radmc3d.write(f,
                              '----------------------------------------------------------------------------')

    # Run radiative transfer for each observed wavelength
    for _scat, _lam_image in zip(options['scattering'],
                                 options['lam_obs_list']):
        # Remove previous radmc output file, if existing
        radmc_out_path = radmcfolder / 'image.out'
        if radmc_out_path.exists():
            radmc_out_path.unlink()

        radmc_call = (
            f"image incl {options['inc']} posang {options['PA'] - 90} npix "
            f"{options['npix']} lambda {_lam_image * 1e4} "
            f"sizeau {2 * options['rout'] / au} setthreads "
            f"{options['threads']}")
        if _scat:
            radmc_call += ' stokes'
        logging.info(radmc_call)
        disklab.radmc3d.radmc3d(
            radmc_call,
            path=radmcfolder,
            executable=str(radmc3d_exec)
        )

        fits_path = radmcfolder.parent / f'{_lam_image * 1e4:.1f}_mu.fits'
        try:
            im_sim = image.readImage(str(radmc_out_path))
            im_sim.writeFits(str(fits_path), dpc=options['distance'],
                             coord=options['coord'])
        except FileNotFoundError:
            warnings.warn("Could not find the radmc output file.")

    return model_path.parent


if __name__ == '__main__':
    model_options = {
        'mstar': 0.8 * M_sun,
        'lstar': 1 * L_sun,
        'tstar': 3810,
        'nr': 400,
        'rin': 0.32 * au,
        'rout': 250 * au,
        'r_c': 70 * au,
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
        'threads': 1,
    }

    model_parameters = [
        0.87754,  # grain size distribution, a**(4-x)
        2.87614,  # max grain size radial distribution exponent
        0.00171,  # grain size distribution, a**(4-x)
        2.87614,  # d2g exp
        0.00171,  # d2g at 70 au
    ]

    disk_model(model_parameters, model_options)

"""
LOOK FOR ALMA IMAGES OF DISKS WITH CAVITIES, EASIER TO SEE PLANETS, FOR JWST
PROPOSAL

Beta Pictoris last week christine chen
"""
