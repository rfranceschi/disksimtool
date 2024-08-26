import subprocess

from setuptools import setup


def check_dependencies():
    try:
        subprocess.check_call(['radmc3d'], stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        raise RuntimeError("Required software 'software_name' is not "
                           "installed.")


check_dependencies()

setup(
    name='disksimtool',
    version='0.1.0',
    description='A set of tool to simulate and fit disk data.',
    author='Riccardo Franceschi',
    author_email='franceschiriccardo93@gmail.com',
    packages=['disksimtool'],
    install_requires=[
        'astropy',
        'dsharp_opac',
        'gofish',
        'numpy',
        'matplotlib',
        'radmc3dPy',
        'scipy',
        'tqdm',
        'ultranest',
    ],
)
