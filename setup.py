#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:43:08 2019

@author: aurelien
"""

from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()
    
setup(name='dataSlicer',
      version='1',
      description='Package for extraction and fitting of intensity profiles',
      long_description = readme(),
      url='',
      packages = find_packages(),
      install_requires = ['pyqtgraph','matplotlib-scalebar', 'tifffile' ],
      author='Aurelien Barbotin',
      author_email='aurelien.barbotin@dtc.ox.ac.uk',
      package_data = {},
      include_package_data = True,
      license='MIT',
      zip_safe=False)
