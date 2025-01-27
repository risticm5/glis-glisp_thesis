#!/usr/bin/env python

__version__ = '0.0.0'
__author__ = 'Marko Ristic'

from setuptools import setup, find_packages

setup(
    name='glis',
    version='0.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=["numpy","scipy","pyswarm","pydoe","numba"],
)
