from setuptools import setup

setup(
   name='disksimtool',
   version='0.1.0',
   description='A set of tool to simulate and fit disk data.',
   author='Riccardo Franceschi',
   author_email='franceschiriccardo93@gmail.com',
   packages=['disksimtool'],
   install_requires=[
      'astropy',
      'disklab',
      'dsharp_opac',
      'numpy',
      'radmc3dPy',
      'scipy',
      'ultranest',
   ],
   setup_requires=['pytest-runner'],
   tests_require=['pytest', 'flake8'],
)
