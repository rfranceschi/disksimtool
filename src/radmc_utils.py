from pathlib import Path

import disklab
import numpy as np


def write_radmc3d(disk2d, lam, path, show_plots=False, nphot=10000000,
                  nphot_scat=100000):
    """
    convert the disk2d object to radmc3d format and write the radmc3d input
    files.

    Parameters
    ----------
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

    Returns
    -------

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
    disklab.radmc3d.write_dust_density(rmcd_temp, fname='dust_temperature.dat',
                                       path=path, mirror=False)
    disklab.radmc3d.write_dust_density(rho, mirror=False, path=path)
    disklab.radmc3d.write_wavelength_micron(lam_mic, path=path)
    disklab.radmc3d.write_opacity(disk2d, path=path)
    disklab.radmc3d.write_radmc3d_input(
        {
            'scattering_mode': 5,
            'scattering_mode_max': 5,
            # was 5 (most realistic scattering), 1 is isotropic
            'nphot': nphot,
            'nphot_scat': nphot_scat,
            'dust_2daniso_nphi': '60',
            'mc_scat_maxtauabs': '5.d0',
        },
        path=path)


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
            zscat = np.fromfile(f, dtype=np.float64, count=n_th * n_f * 6,
                                sep=' ').reshape([6, n_th, n_f], order='F').T
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
    """
    Helper function: reads next line from file but skips comments and empty
    lines.

    Parameters
    ----------
    filehandle
    comments

    Returns
    -------

    """
    line = filehandle.readline()
    while line.startswith(comments) or line.strip() == '':
        line = filehandle.readline()
    return line
