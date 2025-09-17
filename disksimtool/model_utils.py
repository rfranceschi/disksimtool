import warnings
from pathlib import Path
from typing import List

from autologging import traced, logged
import astropy.constants as c
import disklab
import numpy as np
from gofish import imagecube
from matplotlib import pyplot as plt
from scipy.integrate import simpson

import disksimtool.helper_functions as hf

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value

def get_interfaces_from_log_cell_centers(x):
    """
    Returns the cell interfaces for an array of logarithmic
    cell centers.

    Arguments:
    ----------

    x : array
    :   Array of logarithmically spaced cell centers for
        which the interfaces should be calculated

    Output:
    -------

    xi : array
    :    Array of length len(x)+1 containing the grid interfaces
         defined such that 0.5*(xi[i]+xi[i+1]) = xi
    """
    x = np.asarray(x)
    B = x[1] / x[0]
    A = (B + 1) / 2.
    xi = np.append(x / A, x[-1] * B / A)
    return xi

def get_powerlaw_dust_distribution(sigma_d, a_max, q=3.5, na=10, a0=None, a1=None):
    """
    Makes a power-law size distribution up to a_max, normalized to the given surface density.

    Arguments:
    ----------

    sigma_d : array
        dust surface density array

    a_max : array
        maximum particle size array

    Keywords:
    ---------

    q : float | array
        particle size index, n(a) propto a**-q
        if array, it has to have the same length as sigma_d

    na : int
        number of particle size bins

    a0 : float
        minimum particle size

    a1 : float
        maximum particle size

    Returns:
    --------

    a : array
        particle size grid (centers)

    a_i : array
        particle size grid (interfaces)

    sig_da : array
        particle size distribution of size (len(sigma_d), na)
    """

    if a0 is None:
        a0 = a_max.min()

    if a1 is None:
        a1 = 2 * a_max.max()

    nr = len(sigma_d)
    sig_da = np.zeros([nr, na]) + 1e-100

    a_i = np.logspace(np.log10(a0), np.log10(a1), na + 1)
    a = 0.5 * (a_i[1:] + a_i[:-1])

    # we want to turn q into an array if it isn't one already
    q = q * np.ones(nr)

    for ir in range(nr):

        if a_max[ir] <= a0:
            sig_da[ir, 0] = 1
        else:
            i_up = np.where(a_i < a_max[ir])[0][-1]

            # filling all bins that are strictly below a_max

            if q[ir] == 4.0:
                for ia in range(i_up):
                    sig_da[ir, ia] = np.log(a_i[ia + 1] / a_i[ia])

                # filling the bin that contains a_max
                sig_da[ir, i_up] = np.log(a_max[ir] / a_i[i_up])
            else:
                for ia in range(i_up):
                    sig_da[ir, ia] = a_i[ia +
                                         1]**(4 - q[ir]) - a_i[ia]**(4 - q[ir])

                # filling the bin that contains a_max
                sig_da[ir, i_up] = a_max[ir]**(4 - q[ir]) - \
                    a_i[i_up]**(4 - q[ir])

        # normalize

        sig_da[ir, :] = sig_da[ir, :] / sig_da[ir, :].sum() * sigma_d[ir]

    return a, a_i, sig_da

@logged
@traced
def make_disklab2d_model(
        parameters: List[float],
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
    Create a dislkab model and the opacity needed to run a radiative
    transfer calculation for dust emission.

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
        Normalized profile for the gas surface density distribution. If
        None, use a LBP self similar solution.
    show_plots: bool = False

    Returns
    -------

    """
    # The different indices in the parameters list correspond to different
    # physical paramters

    # n(a) = a**(-size_exp)
    size_exp = parameters[0]
    # a_max = amax_coeff * (d.r / (56 * au)) ** (-amax_exp)
    amax_exp = parameters[1]
    amax_coeff = parameters[2]
    d2g_exp = parameters[3]
    d2g_coeff = parameters[4]

    # read some values from the parameters file
    with np.load(opac_fname) as fid:
        a_opac = fid['a']
        rho_s = fid['rho_s']
        n_a = len(a_opac)

    # start with the 1D model
    r_sep = 20 * au
    n_sep = int(0.2 * nr)

    # Create a grid more refined at smaller radii, to logarithmically sample
    # the disk.
    rmod = np.hstack((np.geomspace(rin, r_sep, n_sep + 1)[:-1],
                      np.linspace(r_sep, rout, nr - n_sep)))
    d = disklab.DiskRadialModel(mstar=mstar, lstar=lstar, tstar=tstar,
                                alpha=alpha, rgrid=rmod)
    if profile_funct is None:
        raise NotImplementedError
    else:
        d.sigma = profile_funct(d.r)
        d.compute_mass()
        d.compute_rhomid_from_sigma()

    if d.mass / mstar > 0.2:
        warnings.warn(
            f'Disk mass is unreasonably high: M_disk / Mstar = '
            f'{d.mass / mstar:.2g}')

    # Add the dust, based on the dust-to-gas parameters.
    # Experiment d2g distribution.
    d2g = d2g_coeff * (d.r / r_c) ** (-d2g_exp)
    d2g = np.minimum(d2g, 0.1)
    # We take as scaling radius the edge of the 870 micron image,
    # for simplicity
    a_max = amax_coeff * (d.r / r_c) ** (-amax_exp)

    a_i = get_interfaces_from_log_cell_centers(a_opac)
    # if we change a0 and a1 we have a different grid than a_opac, and the
    # interpolation creates the wrong g parameter
    #   increase the number of grain size in the opac file
    #   OR we take ~150 grain sizes in the opac file and then interpolate 15
    #   grains for radmc3d (change a1 in the next
    #   call). Radmc will still complain though, we would have to
    #   recalculate g.

    a, a_i, sig_da = get_powerlaw_dust_distribution(d.sigma * d2g,
                                                    np.minimum(a_opac[-1],
                                                               a_max),
                                                    q=size_exp,
                                                    na=n_a, a0=a_i[0],
                                                    a1=a_i[-1])

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

        plt.savefig('./model_plots.png')

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
    d.mean_opacity_planck[7:-7] = hf.movingaverage(d.mean_opacity_planck, 10)[
                                  7:-7]
    d.mean_opacity_rosseland[7:-7] = hf.movingaverage(d.mean_opacity_rosseland,
                                                      10)[7:-7]

    if show_plots:
        ax.loglog(d.r / au, d.mean_opacity_planck, 'C0--')
        ax.loglog(d.r / au, d.mean_opacity_rosseland, 'C1--')

        f, ax = plt.subplots()
        ax.loglog(d.r / au, d.tmid)

        ax.set_xlabel('radius [au]')
        ax.set_ylabel(r'T$_{mid}$')

    n_average = 35
    d.compute_disktmid(keeptvisc=False)
    d.tmid = hf.running_average(d.tmid, n=n_average)
    d.compute_hsurf()
    d.hs = hf.running_average(d.hs, n=n_average)
    d.compute_flareindex()
    d.flidx = hf.running_average(d.flidx, n=n_average)
    d.compute_flareangle_from_flareindex(inclrstar=True)
    d.flang = hf.running_average(d.flang, n=n_average)
    d.compute_cs_and_hp()
    d.compute_mean_opacity()

    # iterate the temperature
    if show_plots:
        f, ax = plt.subplots(2, 1, dpi=150, sharex=True)

    n_iter = 100
    for iter in range(n_iter):
        make_disklab2d_model._log.debug(f'1D iteration {iter + 1}')
        tmid_previous = d.tmid
        hs_previous = d.hs
        flidx_previous = d.flidx

        d.compute_hsurf()
        d.hs = hf.running_average(d.hs, n=n_average)
        d.hs = hs_previous + 0.08 * (d.hs - hs_previous)

        d.compute_flareindex()
        d.flidx = hf.running_average(d.flidx, n=n_average)
        d.flidx = flidx_previous + 0.08 * (d.flidx - flidx_previous)
        d.compute_flareangle_from_flareindex(inclrstar=True)
        d.flang = hf.running_average(d.flang, n=n_average)

        d.compute_disktmid(keeptvisc=False)
        d.tmid = hf.running_average(d.tmid, n=n_average)
        d.tmid = tmid_previous + 0.08 * (d.tmid - tmid_previous)

        if all(np.abs(tmid_previous / d.tmid - 1) < 0.01):
            make_disklab2d_model._log.debug('Converged.')
            break
        # else:
        #     print(f"not converged, max change {max(np.abs(tmid_previous /
        #     d.tmid - 1))}")

        d.compute_cs_and_hp()
        d.compute_mean_opacity()

        if show_plots:
            if (iter % 9) == 0:
                ax[0].loglog(d.r / au, d.hs / au, label=iter)
                ax[1].loglog(d.r / au, d.tmid, label=iter)

    d.tmid = hf.running_average(d.tmid, n=n_average)

    if show_plots:
        ax[-1].set_xlim(120, 400)
        ax[0].set_ylim(1e1, 6e1)
        ax[1].set_ylim(1e0, 5e1)
        ax[0].set_title("hs")
        ax[1].set_title("tmid")
        plt.suptitle("Midplane iterations")
        plt.legend()
        plt.show()
        plt.savefig('./midplane_iterations.png')

    # ---- Make a 2D model out of it ----

    make_disklab2d_model._log.debug('Create Disk2D model...')
    disk2d = disklab.Disk2D(
        disk=d,
        meanopacitymodel=d.meanopacitymodel,
        nz=50,
        zrmax=0.5,
    )
    make_disklab2d_model._log.debug('Done.')

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

    make_disklab2d_model._log.debug(len(disk2d.verts))
    for vert in disk2d.verts:
        vert.compute_mean_opacity()
        vert.irradiate_with_flaring_index()

    # disk2d.radial_raytrace()

    n_iter = 20
    for iter in range(n_iter):
        make_disklab2d_model._log.debug('Start ray tracing.')
        disk2d.radial_raytrace()
        make_disklab2d_model._log.debug('Ray tracing done.')
        for i, vert in enumerate(disk2d.verts):
            make_disklab2d_model._log.debug(f'2D iteration {iter + 1}, '
                                            f'vertical lvl {i}.')
            vert.compute_rhogas_hydrostatic()
            vert.rhogas = hf.running_average(vert.rhogas, n=n_average)
            vert.compute_mean_opacity()
            vert.irradiate_with_flaring_index()

            # this line turns viscous heating OFF:
            vert.visc_src = np.zeros_like(vert.z)

            # this line turns viscous heating ON:
            # vert.compute_viscous_heating()

            vert.solve_vert_rad_diffusion()
            vert.tgas = hf.running_average(vert.tgas, n=n_average)
            vert.tgas = (vert.tgas ** 4 + 15 ** 4) ** (1 / 4)
            for dust in vert.dust:
                dust.compute_settling_mixing_equilibrium()

    # --- done setting up the radmc3d model ---
    return disk2d

@traced
def get_profile_from_fits(fname: Path, clip: float =2.5,
                          show_plots: bool = False, inc: float= 0,
                          PA: float = 0, z0: float = 0.0, psi: float = 0.0,
                          beam: tuple = None, r_norm: float = None,
                          norm: float = None, **kwargs):
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
        the scale height at 1 arcse and the radial exponent used in the
        deprojection

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
        raise ValueError('Only norm or r_norm can be set, not both!')

    if isinstance(fname, imagecube):
        data = fname
    else:
        data = imagecube(fname, FOV=clip)

    if beam is not None:
        data.bmaj, data.bmin, data.bpa = beam
        data.beamarea_arcsec = data._calculate_beam_area_arcsec()
        data.beamarea_str = data._calculate_beam_area_str()
    x, y, dy = data.radial_profile(inc=inc, PA=PA, z0=z0, psi=psi, **kwargs)

    if not data.bunit.lower() == 'jy/beam':
        if data.bunit.lower() == 'jy/pixel':
            y *= data.pix_per_beam
            dy *= data.pix_per_beam
        else:
            raise ValueError(
                    'Unknown unit, please implement conversion to Jy/beam here.')

        # if data.bunit.lower() == 'jy/beam':
        #     y *= 1e-23 / data.beamarea_str
        #     dy *= 1e-23 / data.beamarea_str
        # elif data.bunit.lower() == 'jy/pixel':
        #     y *= 1e-23 * data.pix_per_beam / data.beamarea_str
        #     dy *= 1e-23 * data.pix_per_beam / data.beamarea_str
        # else:
        #     raise ValueError(
        #         'Unknown unit, please implement conversion to CGS here.')

    if r_norm is not None:
        norm = np.interp(r_norm, x, y)
        y /= norm
        dy /= norm
    elif norm is not None:
        y /= norm
        dy /= norm

    if show_plots:
        f, ax = plt.subplots()
        ax.semilogy(x, y)
        ax.fill_between(x, y - dy, y + dy, alpha=0.5)

    return x, y, dy, norm

def lbp_profile(r: float, sigma_coeff: float, r_c: float, gamma: float) -> (
        float):
    return sigma_coeff * (r / r_c)**(-gamma) * np.exp(-(r / r_c)**(2 - gamma))

def lbp_profile_with_rim(r: np.array, sigma_coeff: float, r_c: float,
                         gamma: float, r_exp: float, w: float,
                         gamma_exp: float = 3) -> float:
    """
    Lynden-Bell Pringle profile with inner exponential taper.

    Parameters
    ----------
    r: radial position array
    sigma_coeff: surface density at the scaling radius
    r_c: scaling radius
    gamma: power law exponent
    r_exp: inner ring radius
    w: inner ring tapering radius
    gamma_exp: inner ring tapering exponent

    Returns
    -------

    """
    r_dim = r / r_exp
    rim_mask = r_dim < 1
    surface_density = sigma_coeff * (r / r_c) ** (-gamma) * np.exp(
        -(r / r_c) ** (2 - gamma))
    surface_density[rim_mask] *= np.exp(-((1 - r_dim[rim_mask]) / w) ** gamma_exp)
    return surface_density

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

def sigma_with_smooth_transition(r, sigma_exp, r_exp, p1, p2, r_transition, delta_r, w):
    r = np.asarray(r)
    r_dim = r / r_exp

    # Smooth blend of exponent p(r)
    s = 1 / (1 + np.exp(-(r - r_transition) / delta_r))
    p_r = p1 + (p2 - p1) * s

    # Adjust normalization to preserve continuity at r_transition
    f = (r_transition / r_exp)
    norm_adjust = f ** (p_r - p1)

    surface_density = sigma_exp * norm_adjust * r_dim ** -p_r

    # Inner rim tapering
    rim_mask = r_dim < 1
    surface_density[rim_mask] *= np.exp(-((1 - r_dim[rim_mask]) / w) ** 3)

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
