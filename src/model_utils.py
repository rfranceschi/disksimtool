import warnings

import astropy.constants as c
import disklab
import numpy as np
from dipsy import get_powerlaw_dust_distribution
from dipsy.utils import get_interfaces_from_log_cell_centers
from gofish import imagecube
from matplotlib import pyplot as plt

import src.helper_functions as hf

au = c.au.cgs.value
M_sun = c.M_sun.cgs.value
L_sun = c.L_sun.cgs.value
R_sun = c.R_sun.cgs.value


def make_disklab2d_model(
        parameters: list,
        mstar: float,
        lstar: float,
        tstar: float,
        nr: int,
        alpha: float,
        rin: float,
        rout: float,
        # r_c: float,
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

    size_exp = parameters[0]  # n(a) = a**(4-size_exp)
    amax_exp = parameters[1]  # a_max = amax_coeff * (d.r / (56 * au)) ** (-amax_exp)
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

    # Create a grid more refined at smaller radii, to logarithmically sample the disk.
    rmod = np.hstack((np.geomspace(rin, r_sep, n_sep + 1)[:-1], np.linspace(r_sep, rout, nr - n_sep)))
    d = disklab.DiskRadialModel(mstar=mstar, lstar=lstar, tstar=tstar, alpha=alpha, rgrid=rmod)
    if profile_funct is None:
        raise NotImplementedError
    else:
        d.sigma = profile_funct(d.r)
        d.compute_mass()
        d.compute_rhomid_from_sigma()

    if d.mass / mstar > 0.2:
        warnings.warn(f'Disk mass is unreasonably high: M_disk / Mstar = {d.mass / mstar:.2g}')

    # Add the dust, based on the dust-to-gas parameters.
    # Experiment d2g distribution.
    d2g = d2g_coeff * (d.r / (70 * au)) ** (-d2g_exp)
    d2g = np.minimum(d2g, 0.1)
    # We take as scaling radius the edge of the 870 micron image, for simplicity
    a_max = amax_coeff * (d.r / (70 * au)) ** (-amax_exp)

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
    d.mean_opacity_planck[7:-7] = hf.movingaverage(d.mean_opacity_planck, 10)[7:-7]
    d.mean_opacity_rosseland[7:-7] = hf.movingaverage(d.mean_opacity_rosseland, 10)[7:-7]

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
            print(f"iteration to convergence: {iter}")
            break
        # else:
        #     print(f"not converged, max change {max(np.abs(tmid_previous / d.tmid - 1))}")

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
