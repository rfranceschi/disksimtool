import subprocess
import tempfile
import warnings
from pathlib import Path

import dsharp_opac as opacity
import numpy as np
import tqdm
from disklab.radmc3d import write_wavelength_micron
import dsharp_opac as do

from disksimtool import radmc_utils


def read_opacs(fname):
    with np.load(fname) as fid:
        opac_dict = {k: v for k, v in fid.items()}
    return opac_dict


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

        cmd = f'optool -chop {chop} -s {n_angle} -p {porosity} {composition} -a {_a * 0.9e4} {_a * 1.1e4} 3.5 10 -l {lam_str} -radmc'
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
        optool_data = radmc_utils.read_radmc_opacityfile(fname)

        # put data of this particle into the big arrays
        k_abs[ia, :] = optool_data['k_abs']
        k_sca[ia, :] = optool_data['k_sca']
        g[ia, :] = optool_data['g']
        lam = optool_data['lam']

        if scatter:
            theta = optool_data['theta']
            if zscat is None:
                zscat = np.zeros([len(a), len(lam), len(theta), 6])
            zscat[ia, ...] = optool_data['zscat']

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
        'output': output,
        'rho_s': rho_s,
    }

    if scatter:
        output['zscat'] = zscat
        output['theta'] = theta
        output['n_th'] = len(theta)

    return output


def make_opacs(a, lam, fname='dustkappa', porosity=None, constants=None, n_theta=101, optool=True,
               composition='dsharp'):
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

    if optool:
        try:
            rho_s = optool_wrapper([a[0]], lam, chop=5, porosity=porosity, composition=composition)['rho_s']
            optool_available = True
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


def compute_opac(lam_opac, n_a, n_theta, porosity):
    a_opac = np.logspace(-5, 0, n_a)
    composition = 'diana'

    # Make opacities if necessary
    fname = Path('opacities/dustkappa')
    fname.parent.mkdir(exist_ok=True)
    opac_dict = make_opacs(a_opac, lam_opac, fname=str(fname), porosity=porosity, n_theta=n_theta,
                           composition=composition, optool=True)
    fname_opac = opac_dict['filename']

    # This part chops the very-forward scattering part of the phase function.
    # This part is basically the same as no scattering, but are treated by the code as a scattering event.
    # By cutting this part out of the phase function, we avoid those non-scattering scattering events.
    fname_opac_chopped = '_chopped.'.join(fname_opac.rsplit('.', 1))

    k_sca_nochop = opac_dict['k_sca']
    g_nochop = opac_dict['g']

    zscat, zscat_nochop, k_sca, g = chop_forward_scattering(opac_dict)

    opac_dict['k_sca'] = k_sca
    opac_dict['zscat'] = zscat
    opac_dict['g'] = g
    opac_dict['composition'] = composition

    rho_s = opac_dict['rho_s']
    m = 4 * np.pi / 3 * rho_s * a_opac ** 3

    do.write_disklab_opacity(fname_opac_chopped, opac_dict)


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
