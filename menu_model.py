import logging
import pickle
import shutil
import sys
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

logging.basicConfig(level=logging.WARNING)
radmc3d_exec = Path('~/bin/radmc3d').expanduser()

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value


def disk_model(parameters: list, options: dict, show_plots: bool = False,
               models_root: Path = Path('./runs/')) -> (
        Path):
    """

    Parameters
    ----------
    parameters: list
        Free parameters.
    options: list
        Fixed parameters.
    show_plots: bool
    models_root: Path
        Directory where models are stored.
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
        n_lam = 100  # number of wavelength points
        n_a = 50  # number of particle sizes
        n_theta = 181  # number of angles in the scattering phase function
        porosity = 0.3

        # wavelength and particle sizes grids
        lam_opac = np.logspace(-5, 0, n_lam)
        # We insert the observation wavelengths to be sure we don't need
        # interpolation
        for _lam_obs in options['lam_obs_list']:
            ilam = np.abs(lam_opac - _lam_obs).argmin()
            lam_opac[ilam] = _lam_obs

        opac.compute_opac(lam_opac, n_a, n_theta, porosity,
                          fname=options['fname_opac'])
        opac_dict = opac.read_opacs(Path(options['fname_opac']))

    # DISK MODEL
    r = np.linspace(options['rin'], options['rout'], options['nr'])

    profile = options['sigma_funct'](r)
    disk_gas_mass = (integrate_sigma(r, profile) / c.M_sun.cgs.value)
    logging.info(f'Total disk mass: {disk_gas_mass:.2} M_sun')

    model_name = 'model_' + '_'.join([f'{_par:.2e}' for _par in parameters])
    model_path = models_root / model_name / 'model.pkl'

    if model_path.is_file():
        with open(model_path, 'rb') as fff:
            logging.warning(f'Loading model from {model_path}.')
            disk2d = pickle.load(fff)
    else:
        logging.info(f'Writing to {model_name} directory.')
        # Surface density parameters fixed from Menu et al. 2014 (
        # https://arxiv.org/pdf/1402.6597).
        # density_func = partial(sigma_with_rim, **density_params)

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
            options['sigma_funct'],
            show_plots=show_plots
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
            f"image incl {options['inc']} posang {options['PA'] - 90} "
            f"npix {options['npix']} "
            f"lambda {_lam_image * 1e4} "
            f"sizeau {2 * options['rout'] / au} "
            f"setthreads {options['threads']}")
        if _scat:
            radmc_call += ' stokes'
        radmc_call += ' sloppy'
        logging.info(radmc_call)
        disklab.radmc3d.radmc3d(
            radmc_call,
            path=radmcfolder,
            executable=str(radmc3d_exec)
        )

        fits_path = radmcfolder.parent / f'{_lam_image * 1e4:.1f}_mu.fits'
        try:
            im_sim = image.readImage(str(radmc_out_path))
            im_sim.writeFits(str(fits_path), dpc=options['distance_pc'],
                             coord=options['coord'])
        except FileNotFoundError:
            warnings.warn("Could not find the radmc output file.")

    return model_path.parent


if __name__ == '__main__':
    model_options = {'mstar': 0.75 * M_sun, 'lstar': 0.34 * L_sun,
                     'tstar': 3810, 'nr': 250, 'rin': 0.32 * au,
                     'rout': 100 * au, 'r_c': 30 * au, 'alpha': 1e-3,
                     'fname_opac': 'opacities/dustkappa_p30_chopped.npz',
                     'inc': 7, 'PA': 0, 'distance_pc': 56,
                     'lam_obs_list': [0.000165, 0.087],
                     # 'lam_obs_list': [0.000165, 0.0015, 0.087],
                     'scattering': [True, False],
                     # 'scattering': [True, False, False],
                     'coord': '11h01m51.9053285064s -34d42m17.033218380s',
                     'npix': 59, 'threads': 16, 'opac': 0.3}

    # params = {
    #     'sigma_exp': 24,
    #     'r_exp': 3.1 * au,
    #     'p': 0.5,
    #     'w': 0.45,
    # }
    # sigma_funct = partial(sigma_with_rim, **params)

    params = {
        'sigma_exp': 24,
        'r_exp': 3.1 * au,
        'p1': 0.5,
        'p2': 0.5,
        'r_transition': 50 * au,
        'delta_r': 5 * au,  # smoothing width
        'w': 0.45,
    }

    sigma_funct = partial(sigma_with_smooth_transition, **params)
    model_options['sigma_funct'] = sigma_funct

    # p_0 = np.linspace(0.3, 0.7, 5)
    # p_1 = np.linspace(4, 6, 4)
    # p_2 = np.linspace(0.8, 1.2, 4)
    # p_3 = np.linspace(4, 6, 4)
    # p_4 = np.logspace(-1.3, -0.7, 4)
    #
    # P_0, P_1, P_2, P_3, P_4 = np.asarray(np.meshgrid(p_0, p_1, p_2, p_3, p_4))

    default_params = np.array([0.3550639294858283, 2.786764286453357, 0.17449244727854488,
         3.409432193743408, 0.026388837407451893])

    param_index = int(0)
    param_sample_size = 5
    edges = (0, 6)
    param_sample = edges[0] + np.random.rand(param_sample_size) * np.abs(edges[1] - edges[0])

    params_list = [
        [0.9470830780128577, 6.250400805278292, 0.4495865608589366,
         2.714376852103456, 0.012578747337948015],
        [0.8973417633901676, 5.224821013634942, 0.2682925040778293,
         3.1160551239317567, 0.010937911098339989],
        [0.35437629358984446, 7.529230900288666, 0.8028574469791965,
         1.098347051404548, 0.010833302865208646],
        [0.7034638956083591, 5.313626061198816, 0.3341984513082757,
         2.7751846814896943, 0.011149992504514802],
        [0.8511748188228364, 6.8050370760185075, 0.430174779861027,
         3.335803567498621, 0.010681682412816041],
        [0.5852191218190014, 6.846325263173142, 0.6740089047435488,
         2.5768083234735704, 0.011240621973327876],
        [0.285496370974331, 7.182494704289713, 0.6597999916826546,
         1.1529245892174511, 0.011625326773376682],
        [0.866794742028872, 1.2137567999054988, 0.6934175276847655,
         0.7058579661788877, 0.012024548501282782],
        [0.6782911726426762, 6.859929532883598, 0.40350748244067036,
         2.5027585671259533, 0.012931388469342945],
        [0.8810030211918097, 6.921029928805781, 0.6836746864091748,
         3.8949346918876158, 0.01771723921404255],
    ]


    for i, _params in enumerate(
            params_list
            # [default_params,]
            # [[2.85e-01, 7.18e+00, 6.60e-01, 1.15e+00, 1.16e-02],] # manual tune sigma outer slope
    ):
        try:
            model_dir = disk_model(_params, model_options,
                                   models_root = Path('./runs_best_cluster'))
            # shutil.rmtree(model_dir / 'radmc_run')
            with open(model_dir / 'model_info.txt', "w") as file:
                file.write(f"Model parameters:   {_params}\n")

            target_dir = model_dir.parent
            # target_dir = model_dir.parent / f'test_p{param_index}'
            # target_dir.mkdir(parents=True, exist_ok=True)
            model_dir.rename(target_dir / ('' +
                                           model_dir.name))

        except OSError as e:
            print(e)
            continue