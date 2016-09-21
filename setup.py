# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 13:21:02 2016

@author: Shubhankar Mitra
"""

from setuptools import setup

setup(name='ParametricModeling',
      version='0.1.9',
      description='An Sklearn interface around scipy.optimize.least_squares for parametric modeling',
      url='http://github.com/shubhankar90/ParametricModeling',
      author='Shubhankar Mitra',
      author_email='shubhankar90@gmail.com',
      license='BSD 3 clause',
      packages=['ParametricModeling'],
      install_requires=[
          'scipy','numpy','sklearn'
      ],
      zip_safe=False)