import warnings
import pickle
from pathlib import Path
import tempfile
import subprocess

import matplotlib.pyplot as plt
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
from scipy.integrate import simpson
from scipy.interpolate import interp1d

from helper_functions_old import *


au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value


def make_disklab2d_model(
        # parameters: list,
        a_obs: float,
        mstar: float,
        lstar: float,
        tstar: float,
        nr: int,
        alpha: float,
        rin: float,
        rout: float,
        # r_c: float,
        opac_fname: str,
        profile_funct_dust: callable,
        profile_funct_gas: callable = None,
        show_plots: bool = False
):
    """
    Create a dislkab model and the opacity needed to run a radiative transfer calculation for dust emission of a single
    dust population.

    Parameters
    ----------
    parameters: list
        Additional parameters for the disk gas and dust distribution.
    d2g: float
        Dust-to-gas ratio of the single dust population
    a_obs: float
        Emitting grain size.
    mstar: float
    lstar: float
    tstar: float
    nr: int
    alpha: float
    rin: float
    rout: float
    r_c: float
    opac_fname: str
    profile_funct_dust: callable
        Normalized profile for the dust population surface density distribution.
    profile_funct_gas: callable
        Normalized profile for the gas surface density distribution. If None, use a LBP self similar solution.
    show_plots: bool = False

    Returns
    -------

    """
    # The different indices in the parameters list correspond to different physical paramters

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
    if profile_funct_gas is None:
        raise NotImplementedError
        # if parameters is None:
        #     raise ValueError('You must provide the values needed to compute a LBS soltion if not using a custom'
        #                      'profile for the gas surface density distribution.')
        # d.make_disk_from_simplified_lbp(sigma_coeff, r_c, sigma_exp)
    else:
        d.sigma = profile_funct_gas(d.r)
        d.compute_mass()
        d.compute_rhomid_from_sigma()

    if d.mass / mstar > 0.2:
        warnings.warn(f'Disk mass is unreasonably high: M_disk / Mstar = {d.mass / mstar:.2g}')

    # Add the dust.
    sigma_dust = profile_funct_dust(d.r)
    index = np.nonzero(a_obs <= a_opac)[0][0]
    d.add_dust(agrain=a_opac[index], xigrain=rho_s, dtg=sigma_dust / d.sigma)

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

    # compute the mean opacities_IMLup
    d.meanopacitymodel = ['dustcomponents', {'method': 'simplemixing'}]
    d.compute_mean_opacity()

    if show_plots:
        f, ax = plt.subplots()
        ax.loglog(d.r / au, d.mean_opacity_planck, label='mean plack')
        ax.loglog(d.r / au, d.mean_opacity_rosseland, label='mean rosseland')

        ax.set_xlabel('radius [au]')
        ax.set_ylabel('mean opacity')
        ax.legend()

    # smooth the mean opacities_IMLup
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


def interp_profile(profile_dict: dict, total_disk_mass: float) -> callable:
    """
    Interpolate a profile to use as gas surface density, normalized to a given total disk mass.

    Parameters
    ----------
    profile_dict: dict
        Dictionary containing the profile to use as surface density distribution. The keys must be named 'x' and 'y'.
    total_disk_mass: float
        Total disk mass.

    Returns
    -------
    float
        Normalized gas surface density function.
    """
    normalization_constant = total_disk_mass / simpson(2 * np.pi * profile_dict['x'] * profile_dict['y'],
                                                       profile_dict['x'])

    # The profile cannot be <= 0, make sure that doesn't happen.
    y_profile = np.where(profile_dict['y'] > 0, profile_dict['y'], 0)
    # y_profile = np.copy(profile_dict['y'])
    # if np.min(y_profile) <= 0:
    #     y_profile += np.max(y_profile)

    normalization_constant = total_disk_mass / simpson(2 * np.pi * profile_dict['x'] * y_profile, profile_dict['x'])

    interp = interp1d(profile_dict['x'], y_profile, fill_value='extrapolate', kind='quadratic')
    return lambda x: normalization_constant * interp(x)


def get_profile_from_fits(fname, clip=2.5, show_plots=False, inc=0, PA=0, z0=0.0, psi=0.0, beam=None, r_norm=None,
                          norm=None, **kwargs):
    """Get radial profile from fits file.

    Reads a fits file and determines a radial profile with `imagecube`

    fname : str | Path
        path to fits file

    clip : float
        clip the image at that many image units (usually arcsec)

    show_plots : bool
        if true: produce some plots for sanity checking

    inc, PA : float
        inclination and position angle used in the radial profile

    z0, psi : float
        the scale height at 1 arcse and the radial exponent used in the deprojection

    beam : None | tuple
        if None: will be determined by imgcube
        if 3-element tuple: assume this beam a, b, PA.

    r_norm : None | float
        if not None: normalize at this radius

    norm : None | float
        divide by this norm

    kwargs are passed to radial_profile

    Returns:
    x, y, dy: arrays
        radial grid, intensity (cgs), error (cgs)
    """

    if norm is not None and r_norm is not None:
        raise ValueError('only norm or r_norm can be set, not both!')

    if isinstance(fname, imagecube):
        data = fname
    else:
        data = imagecube(fname, FOV=clip)

    if beam is not None:
        data.bmaj, data.bmin, data.bpa = beam
        data.beamarea_arcsec = data._calculate_beam_area_arcsec()
        data.beamarea_str = data._calculate_beam_area_str()

    x, y, dy = data.radial_profile(inc=inc, PA=PA, z0=z0, psi=psi, **kwargs)

    if data.bunit.lower() == 'jy/beam':
        y *= 1e-23 / data.beamarea_str
        dy *= 1e-23 / data.beamarea_str
    elif data.bunit.lower() == 'jy/pixel':
        y *= 1e-23 * data.pix_per_beam / data.beamarea_str
        dy *= 1e-23 * data.pix_per_beam / data.beamarea_str
    else:
        raise ValueError('unknown unit, please implement conversion to CGS here')

    if r_norm is not None:
        norm = np.interp(r_norm, x, y)
        y /= norm
        dy /= norm

    if norm is not None:
        y /= norm
        dy /= norm

    if show_plots:
        f, ax = plt.subplots()
        ax.semilogy(x, y)
        ax.fill_between(x, y - dy, y + dy, alpha=0.5)

    return x, y, dy
