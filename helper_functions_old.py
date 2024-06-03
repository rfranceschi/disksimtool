import logging
import warnings
import pickle
from pathlib import Path
import tempfile
import subprocess

import matplotlib.pyplot as plt
import optool
from astropy.modeling.functional_models import Exponential1D
from matplotlib import lines, text
from matplotlib.colors import Normalize
import numpy as np
import tqdm

from astropy.modeling.powerlaws import PowerLaw1D
import astropy.constants as c
from astropy import units as u
from gofish import imagecube

import dsharp_opac as opacity
from dipsy import get_powerlaw_dust_distribution
from dipsy.utils import get_interfaces_from_log_cell_centers
from disklab.radmc3d import write_wavelength_micron
import disklab

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value


def chop_forward_scattering(opac_dict, chopforward=5):
    """Chop the forward scattering.

    This part chops the very-forward scattering part of the phase function.
    This very-forward scattering part is basically the same as no scattering,
    but is treated by the code as a scattering event. By cutting this part out
    of the phase function, we avoid those non-scattering scattering events.
    This needs to recalculate the scattering opacity kappa_sca and asymmetry
    factor g.

    Parameters
    ----------
    opac_dict : dict
        opacity dictionary as produced by dsharp_opac

    chopforward : float
        up to which angle to we truncate the forward scattering
    """

    k_sca = opac_dict['k_sca']
    theta = opac_dict['theta']
    g = opac_dict['g']
    rho_s = opac_dict['rho_s']
    lam = opac_dict['lam']
    a = opac_dict['a']
    m = 4 * np.pi / 3 * rho_s * a ** 3

    n_a = len(a)
    n_lam = len(lam)

    if 'zscat' in opac_dict:
        zscat = opac_dict['zscat']
    else:
        S1 = opac_dict['S1']
        S2 = opac_dict['S2']
        zscat = opacity.calculate_mueller_matrix(lam, m, S1, S2)['zscat']

    zscat_nochop = zscat.copy()

    mu = np.cos(theta * np.pi / 180.)
    # dmu = np.diff(mu)

    for grain in range(n_a):
        for i in range(n_lam):
            if chopforward > 0:
                iang = np.where(theta < chopforward)
                if theta[0] == 0.0:
                    iiang = np.max(iang) + 1
                else:
                    iiang = np.min(iang) - 1
                zscat[grain, i, iang, :] = zscat[grain, i, iiang, :]

                # zav = 0.5 * (zscat[grain, i, 1:, 0] + zscat[grain, i, :-1, 0])
                # dum = -0.5 * zav * dmu
                # integral = dum.sum() * 4 * np.pi
                # k_sca[grain, i] = integral

                # g = <mu> = 2 pi / kappa * int(Z11(mu) mu dmu)
                # mu_av = 0.5 * (mu[1:] + mu[:-1])
                # P_mu = 2 * np.pi / k_sca[grain, i] * 0.5 * (zscat[grain, i, 1:, 0] + zscat[grain, i, :-1, 0])
                # g[grain, i] = np.sum(P_mu * mu_av * dmu)

                k_sca[grain, i] = -2 * np.pi * np.trapz(zscat[grain, i, :, 0], x=mu)
                g[grain, i] = -2 * np.pi * np.trapz(zscat[grain, i, :, 0] * mu, x=mu) / k_sca[grain, i]

    return zscat, zscat_nochop, k_sca, g


def make_opacs(a, lam, fname='dustkappa', porosity=None, constants=None, n_theta=101, optool=True, composition='dsharp'):
    """make optical constants file"""

    if n_theta // 2 == n_theta / 2:
        n_theta += 1
        print(f'n_theta needs to be odd, will set it to {n_theta}')

    n_a = len(a)
    n_lam = len(lam)

    if (composition.lower() != 'dsharp') and (optool is False):
        raise ValueError('non dsharp opacities, should use optool')

    if constants is None:
        if porosity is None:
            porosity = 0.0

        if porosity < 0.0 or porosity >= 1.0:
            raise ValueError('porosity has to be >=0 and <1!')

        if porosity > 0.0:
            fname = fname + f'_p{100 * porosity:.0f}'

        if not optool:
            constants = opacity.get_dsharp_mix(porosity=porosity)
    else:
        if porosity is not None:
            raise ValueError('if constants are given, porosity keyword cannot be used')

    opac_fname = Path(fname).with_suffix('.npz')

    print('test')
    if optool:
        print('test1')
        rho_s = optool_wrapper([a[0]], lam, chop=5, porosity=porosity, composition=composition)['rho_s']
        try:
            rho_s = optool_wrapper([a[0]], lam, chop=5, porosity=porosity, composition=composition)['rho_s']
            optool_available = True
            print('test2')
        except FileNotFoundError:
            warnings.warn('optool unavailable, cannot check rho_s to be consistent or recalculate opacities this way')
            optool_available = False
            rho_s = None
    else:
        diel_const, rho_s = constants

    run_opac = True

    if opac_fname.is_file():

        opac_dict = read_opacs(opac_fname)
        # if all the grids agree ...
        run_opac = False
        if len(opac_dict['a']) != n_a:
            print('n_a does not match')
            run_opac = True
        if not np.allclose(opac_dict['a'], a):
            print('a grid not identical')
            run_opac = True
        if len(opac_dict['lam']) != n_lam:
            print('n_lam not identical')
            run_opac = True
        if not np.allclose(opac_dict['lam'], lam):
            print('lambda grid not identical')
            run_opac = True
        if len(opac_dict['theta']) != n_theta:
            print(f'n_theta in dict ({len(opac_dict["theta"])}) != {n_theta}')
            run_opac = True

        if ('composition' in opac_dict) and (opac_dict['composition'] != composition):
            print(f'composition in dict ({opac_dict["composition"]}) != {composition}')
            run_opac = True

        # if optool is used and available, or we use dsharp: then we compare densities
        if (not optool or (optool and optool_available)) and (opac_dict['rho_s'] != rho_s):
            run_opac = True

        # if we need to run it and optool is used but unavailable: ERROR
        if run_opac and optool and not optool_available:
            raise FileNotFoundError('optool unavailable')

    if not run_opac:
        # ... then we don't calculate opacities!
        print(f'using opacities from file {opac_fname}')
    else:
        if optool:
            if optool_available:
                print('using optool: ')
                opac_dict = optool_wrapper(a, lam, n_angle=n_theta - 1, composition=composition)
            else:
                raise FileNotFoundError('optool unavailable, cannot calculate opacities!')
        else:
            # call the Mie calculation & store the opacity in a npz file
            opac_dict = opacity.get_smooth_opacities(
                a,
                lam,
                rho_s=rho_s,
                diel_const=diel_const,
                extrapolate_large_grains=False,
                n_angle=(n_theta + 1) // 2)

        print(f'writing opacity to {opac_fname} ... ', end='', flush=True)
        opacity.write_disklab_opacity(opac_fname, opac_dict)
        print('Done!')

    opac_dict['filename'] = str(opac_fname)

    return opac_dict

def write_radmc3d(disk2d, lam, path, show_plots=False, nphot=10000000, nphot_scat=100000):
    """
    convert the disk2d object to radmc3d format and write the radmc3d input files.

    disk2d : disklab.Disk2D instance
        the disk

    lam : array
        wavelength grid [cm]

    path : str | path
        the path into which to write the output

    show_plots : bool
        if true: produce some plots for checking

    nphot : int
        number of photons to send
    """

    rmcd = disklab.radmc3d.get_radmc3d_arrays(disk2d, showplots=show_plots)

    # Assign the radmc3d data

    ri = rmcd['ri']
    thetai = rmcd['thetai']
    phii = rmcd['phii']
    rho = rmcd['rho']
    n_a = rho.shape[-1]

    # we need to tile this for each species

    rmcd_temp = rmcd['temp'][:, :, None] * np.ones(n_a)[None, None, :]

    # Define the wavelength grid for the radiative transfer

    lam_mic = lam * 1e4

    # Write the `RADMC3D` input

    disklab.radmc3d.write_stars_input(disk2d.disk, lam_mic, path=path)
    disklab.radmc3d.write_grid(ri, thetai, phii, mirror=False, path=path)
    disklab.radmc3d.write_dust_density(rmcd_temp, fname='dust_temperature.dat', path=path, mirror=False)
    disklab.radmc3d.write_dust_density(rho, mirror=False, path=path)
    disklab.radmc3d.write_wavelength_micron(lam_mic, path=path)
    disklab.radmc3d.write_opacity(disk2d, path=path)
    disklab.radmc3d.write_radmc3d_input(
        {
            'scattering_mode': 5,
            'scattering_mode_max': 5,  # was 5 (most realistic scattering), 1 is isotropic
            'nphot': nphot,
            'nphot_scat': nphot_scat,
            'dust_2daniso_nphi': '60',
            'mc_scat_maxtauabs': '5.d0',
        },
        path=path)

def read_opacs(fname):
    with np.load(fname) as fid:
        opac_dict = {k: v for k, v in fid.items()}
    return opac_dict


def running_average(a, n=1):
    b = np.concatenate((np.ones(n) * a[0], a, np.ones(n) * a[-1]))
    return np.convolve(b, np.ones(2 * n + 1) / (2 * n + 1), mode='valid')


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def make_disklab2d_model(
        parameters: list,
        mstar: float,
        lstar: float,
        tstar: float,
        nr: int,
        alpha: float,
        rin: float,
        rout: float,
        r_c: float,
        opac_fname: str,
        profile_funct: callable = None,
        show_plots: bool = False
):
    """
    Create a dislkab model and the opacity needed to run a radiative transfer calculation for dust emission.

    Parameters
    ----------
    parameters: list
        Additional parameters for the disk gas and dust distribution.
    mstar: float
    lstar: float
    tstar: float
    nr: int
    alpha: float
    rin: float
    rout: float
    r_c: float
    opac_fname: str
    profile_funct: callable
        Normalized profile for the gas surface density distribution. If None, use a LBP self similar solution.
    show_plots: bool = False

    Returns
    -------

    """
    # The different indices in the parameters list correspond to different physical paramters

    size_exp = parameters[0]
    amax_exp = parameters[1]
    amax_coeff = parameters[2]
    # cutoff_exp_amax = parameters[3]
    # cutoff_r = parameters[4]

    # hard-coded gas parameters
    sigma_coeff = 28.4
    sigma_exp = 1.0

    # read some values from the parameters file

    with np.load(opac_fname) as fid:
        a_opac = fid['a']
        rho_s = fid['rho_s']
        n_a = len(a_opac)

    # start with the 1D model

    r_sep = 20 * au
    n_sep = 40

    # Create a grid more refined at smaller radii, to logarithmically sample the disk.
    rmod = np.hstack((np.geomspace(rin, r_sep, n_sep + 1)[:-1], np.linspace(r_sep, rout, nr - n_sep)))
    d = disklab.DiskRadialModel(mstar=mstar, lstar=lstar, tstar=tstar, alpha=alpha, rgrid=rmod)
    if profile_funct is None:
        raise NotImplementedError
        # if parameters is None:
        #     raise ValueError('You must provide the values needed to compute a LBS soltion if not using a custom'
        #                      'profile for the gas surface density distribution.')
        # d.make_disk_from_simplified_lbp(sigma_coeff, r_c, sigma_exp)
    else:
        d.sigma = profile_funct(d.r)
        d.compute_mass()
        d.compute_rhomid_from_sigma()


    if d.mass / mstar > 0.2:
        warnings.warn(f'Disk mass is unreasonably high: M_disk / Mstar = {d.mass / mstar:.2g}')

    # Add the dust, based on the dust-to-gas parameters.

    # Experiment d2g distribution.
    d2g = 0.1
    # We take as scaling radius the edge of the 870 micron image, for simplicity
    a_max = amax_coeff * (d.r / (56 * au)) ** (-amax_exp)

    a_i = get_interfaces_from_log_cell_centers(a_opac)
    # if we change a0 and a1 we have a different grid than a_opac, and the interpolation creates the wrong g parameter
    #   increase the number of grain size in the opac file
    #   OR we take ~150 grain sizes in the opac file and then interpolate 15 grains for radmc3d (change a1 in the next
    #   call). Radmc will still complain though, we would have to recalculate g.
    a, a_i, sig_da = get_powerlaw_dust_distribution(d.sigma * d2g, np.minimum(a_opac[-1], a_max), q=4 - size_exp,
                                                    na=n_a, a0=a_i[0], a1=a_i[-1])

    for _sig, _a in zip(np.transpose(sig_da), a_opac):
        d.add_dust(agrain=_a, xigrain=rho_s, dtg=_sig / d.sigma)

    if show_plots:
        f, ax = plt.subplots()

        ax.contourf(d.r / au, a_opac, np.log10(sig_da.T))

        ax.loglog(d.r / au, a_max, label='a_max')
        ax.loglog(d.r / au, d2g, label='d2g')

        ax.set_xlabel('radius [au]')
        ax.set_ylabel('grain size [cm]')
        ax.set_ylim(1e-5, 1e0)
        ax.legend()

    # load the opacity from the previously calculated opacity table
    for dust in d.dust:
        dust.grain.read_opacity(str(opac_fname))

    # compute the mean opacities
    d.meanopacitymodel = ['dustcomponents', {'method': 'simplemixing'}]
    d.compute_mean_opacity()

    if show_plots:
        f, ax = plt.subplots()
        ax.loglog(d.r / au, d.mean_opacity_planck, label='mean plack')
        ax.loglog(d.r / au, d.mean_opacity_rosseland, label='mean rosseland')

        ax.set_xlabel('radius [au]')
        ax.set_ylabel('mean opacity')
        ax.legend()

    # smooth the mean opacities
    d.mean_opacity_planck[7:-7] = movingaverage(d.mean_opacity_planck, 10)[7:-7]
    d.mean_opacity_rosseland[7:-7] = movingaverage(d.mean_opacity_rosseland, 10)[7:-7]

    if show_plots:
        ax.loglog(d.r / au, d.mean_opacity_planck, 'C0--')
        ax.loglog(d.r / au, d.mean_opacity_rosseland, 'C1--')

        f, ax = plt.subplots()
        ax.loglog(d.r / au, d.tmid)

        ax.set_xlabel('radius [au]')
        ax.set_ylabel(r'T$_{mid}$')

    n_average = 35
    d.compute_disktmid(keeptvisc=False)
    d.tmid = running_average(d.tmid, n=n_average)
    d.compute_hsurf()
    d.hs = running_average(d.hs, n=n_average)
    d.compute_flareindex()
    d.flidx = running_average(d.flidx, n=n_average)
    d.compute_flareangle_from_flareindex(inclrstar=True)
    d.flang = running_average(d.flang, n=n_average)
    d.compute_cs_and_hp()
    d.compute_mean_opacity()

    # iterate the temperature
    if show_plots:
        f, ax = plt.subplots(2, 1, dpi=150, sharex=True)

    n_iter = 100
    for iter in range(n_iter):
        tmid_previous = d.tmid
        hs_previous = d.hs
        flidx_previous = d.flidx

        d.compute_hsurf()
        d.hs = running_average(d.hs, n=n_average)
        d.hs = hs_previous + 0.08 * (d.hs - hs_previous)

        d.compute_flareindex()
        d.flidx = running_average(d.flidx, n=n_average)
        d.flidx = flidx_previous + 0.08 * (d.flidx - flidx_previous)
        d.compute_flareangle_from_flareindex(inclrstar=True)
        d.flang = running_average(d.flang, n=n_average)

        d.compute_disktmid(keeptvisc=False)
        d.tmid = running_average(d.tmid, n=n_average)
        d.tmid = tmid_previous + 0.08 * (d.tmid - tmid_previous)

        if all(np.abs(tmid_previous / d.tmid - 1) < 0.01):
            print(f"iteration to convergence: {iter}")
            break
        # else:
        #     print(f"not converged, max change {max(np.abs(tmid_previous / d.tmid - 1))}")

        d.compute_cs_and_hp()
        d.compute_mean_opacity()

        if show_plots:
            if (iter % 9) == 0 :
                ax[0].loglog(d.r / au, d.hs / au, label=iter)
                ax[1].loglog(d.r / au, d.tmid, label=iter)

    d.tmid = running_average(d.tmid, n=n_average)

    if show_plots:
        ax[-1].set_xlim(120, 400)
        ax[0].set_ylim(1e1, 6e1)
        ax[1].set_ylim(1e0, 5e1)
        ax[0].set_title("hs")
        ax[1].set_title("tmid")
        plt.suptitle("Midplane iterations")
        plt.legend()
        plt.show()

    # ---- Make a 2D model out of it ----

    disk2d = disklab.Disk2D(
        disk=d,
        meanopacitymodel=d.meanopacitymodel,
        nz=100,
        zrmax=0.5,
    )

    # taken from snippet vertstruc 2d_1
    # for vert in disk2d.verts:
    #     vert.iterate_vertical_structure()
    # disk2d.radial_raytrace()
    # for vert in disk2d.verts:
    #     vert.solve_vert_rad_diffusion()
    #     vert.tgas = (vert.tgas ** 4 + 15 ** 4) ** (1 / 4)
    #     for dust in vert.dust:
    #         dust.compute_settling_mixing_equilibrium()

    # our own vertical structure, here we turn of viscous heating

    for vert in disk2d.verts:
        vert.compute_mean_opacity()
        vert.irradiate_with_flaring_index()

    disk2d.radial_raytrace()

    n_iter = 20
    for iter in range(n_iter):
        disk2d.radial_raytrace()
        for i, vert in enumerate(disk2d.verts):
            vert.compute_rhogas_hydrostatic()
            vert.rhogas = running_average(vert.rhogas, n=n_average)
            vert.compute_mean_opacity()
            vert.irradiate_with_flaring_index()

            # this line turns viscous heating OFF:
            vert.visc_src = np.zeros_like(vert.z)

            # this line turns viscous heating ON:
            # vert.compute_viscous_heating()

            vert.solve_vert_rad_diffusion()
            vert.tgas = running_average(vert.tgas, n=n_average)
            vert.tgas = (vert.tgas ** 4 + 15 ** 4) ** (1 / 4)
            for dust in vert.dust:
                dust.compute_settling_mixing_equilibrium()

    # --- done setting up the radmc3d model ---
    return disk2d


def read_radmc_opacityfile(file):
    """reads RADMC-3D opacity files, returns dictionary with its contents."""
    file = Path(file)

    if 'dustkapscatmat' in file.name:
        scatter = True

    name = '_'.join(file.stem.split('_')[1:])

    if not file.is_file():
        raise FileNotFoundError(f'file not found: {file}')

    with open(file, 'r') as f:
        iformat = int(get_line(f))
        if iformat == 2:
            ncol = 3
        elif iformat in [1, 3]:
            ncol = 4
        else:
            raise ValueError('Format of opacity file unknown')
        n_f = int(get_line(f))

        # read also number of angles for scattering matrix
        if scatter:
            n_th = int(get_line(f))

        # read wavelength, k_abs, k_sca, g
        data = np.fromfile(f, dtype=np.float64, count=n_f * ncol, sep=' ')

        # read angles and zscat
        if scatter:
            theta = np.fromfile(f, dtype=np.float64, count=n_th, sep=' ')
            zscat = np.fromfile(f, dtype=np.float64, count=n_th * n_f * 6, sep=' ').reshape([6, n_th, n_f], order='F').T
            # zscat = np.moveaxis(zscat, 0, 1)

    data = data.reshape(n_f, ncol)
    lam = 1e-4 * data[:, 0]
    k_abs = data[:, 1]
    k_sca = data[:, 2]

    if iformat in [1, 3]:
        opac_gsca = 1.0 * data[:, 3]

    # define the output

    output = {
        'lam': lam,
        'k_abs': k_abs,
        'k_sca': k_sca,
        'name': name,
    }

    if iformat in [1, 3]:
        output['g'] = opac_gsca
        if scatter:
            output['theta'] = theta
            output['n_th'] = n_th
            output['zscat'] = zscat

    return output


def get_line(filehandle, comments=('=', '#')):
    "helper function: reads next line from file but skips comments and empty lines"
    line = filehandle.readline()
    while line.startswith(comments) or line.strip() == '':
        line = filehandle.readline()
    return line


def optool_wrapper(a, lam, chop=5, porosity=0.3, n_angle=180, composition='dsharp'):
    """
    Wrapper for optool to calculate DSHARP opacities in RADMC-3D format.

    Parameters
    ----------
    a : array
        particle size array
    lam : array | str
        either a string pointing to a RADMC-3d wavelength file or 3-elements: min & max & number of wavelengths
    chop : float, optional
        below how many degrees to chop forward scattering peak, by default 5
    porosity : float, optional
        grain porosity, by default 0.3

    composition : str
        if 'dsharp': use opacities similar to DSHARP (not identical, need to check why)
        if 'diana': use DIANA opacities
        else: use whatever is given as parameters for optool

    Returns
    -------
    dict

    """
    td = None
    if isinstance(lam, str):
        lam_str = lam
        nlam = int(np.fromfile(lam_str, count=1, dtype=int, sep=' '))
    elif len(lam) == 3:
        print('assuming lam specifies minimum wavelength, maximum wavelength, and number of points')
        nlam = lam[3]
        lam_str = '%e %e %d' % tuple(lam)
    elif len(lam) > 3:
        print('assuming lam to be given wavelength grid')
        td = tempfile.TemporaryDirectory()
        write_wavelength_micron(lam_mic=lam * 1e4, path=td.name)
        nlam = len(lam)
        lam_str = str(Path(td.name) / 'wavelength_micron.inp')
    else:
        raise ValueError('lam needs to be a file or of length 3 (lmin, lmax, nl)')

    if composition is None:
        composition = 'dsharp'

    if composition.lower() == 'dsharp':
        composition = '-mie -c h2o-w 0.2 -c astrosil 0.3291 -c fes 0.0743 -c c-org 0.3966'
    elif composition.lower() == 'diana':
        composition = ''

    # initialize arrays

    k_abs = np.zeros([len(a), nlam])
    k_sca = np.zeros_like(k_abs)
    g = np.zeros_like(k_abs)
    zscat = None
    rho_s = None

    # start reading

    for ia, _a in tqdm.tqdm(enumerate(a), total=len(a)):
        # TODO: this next part doesnt work, there is nothing in result.stdout. We should also use the optool python
        #  module instead of running from the command line.

        cmd = f'optool -chop {chop} -s {n_angle} -p {porosity} {composition} -a {_a * 0.9e4} {_a * 1.1e4} 3.5 10 -l {lam_str} -radmc'
        # result = subprocess.run(cmd.split(), capture_output=True)
        result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode()

        if output.split()[-1] == 'dustkapscatmat.inp':
            scatter = True
            fname = 'dustkapscatmat.inp'
        elif output.split()[-1] == 'dustkappa.inp':
            scatter = False
            fname = 'dustkappa.inp'
        else:
            raise ValueError(output.split()[-1])

        # read data, remove file
        optool_data = read_radmc_opacityfile(fname)

        # put data of this particle into the big arrays
        k_abs[ia, :] = optool_data['k_abs']
        k_sca[ia, :] = optool_data['k_sca']
        g[ia, :] = optool_data['g']
        lam = optool_data['lam']

        # TODO
        if scatter:
            theta = optool_data['theta']
            if zscat is None:
                zscat = np.zeros([len(a), len(lam), len(theta), 6])
            zscat[ia, ...] = optool_data['zscat']

        # TODO
        if rho_s is None:
            with open(fname, 'r') as f:
                scat_file = f.read()
            lines = [line for line in scat_file.split('\n') if line.strip('# ').startswith('core')]
            # lines = [line for line in output.split('\n') if line.strip().startswith('core')]
            fractions = np.array([[float(f) for f in line.split()[2:4]] for line in lines])
            rho_s = fractions.prod(axis=1).sum() * (1.0 - porosity)
        Path(fname).unlink()

    if td is not None:
        td.cleanup()

    output = {
        'a': a,
        'lam': lam,
        'k_abs': k_abs,
        'k_sca': k_sca,
        'g': g,
        # TODO
        'output': output,
        'rho_s': rho_s,
    }

    if scatter:
    # if result.scat:
        output['zscat'] = zscat
        output['theta'] = theta
        output['n_th'] = len(theta)

    return output
