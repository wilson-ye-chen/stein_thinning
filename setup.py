from setuptools import setup

setup(
    name='stein_thinning',
    version='0.1',
    description='Optimally compress sampling algorithm outputs',
    url='https://github.com/wilson-ye-chen/stein_thinning',
    author='Stein Thinning team',
    license='MIT',
    packages=['stein_thinning'],
    scripts=['stein_thinning/demo.py'],
    package_data={'stein_thinning': ['sample_chains/gmm/*.csv']},
    install_requires=['jax']
    )
