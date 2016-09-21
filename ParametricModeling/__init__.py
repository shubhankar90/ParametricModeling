# -*- coding: utf-8 -*-

# Authors: Shubhankar Mitra (shubhankar90@gmail.com)
#
# Licence: BSD 3 clause
"""
This project is a sklearn interface wrapper around scipy.optimize.least_squares.
least_squares can be used to solve a nonlinear least-squares problem with bounds 
on the variables.
The aim of this project is to enclose scipy.optimize.least_squares 
function in a sklearn type interface for easier usage and ensure compatibility 
with sklern tools like grid_search.
"""

from .ParametricModeling import ParametricModeling

__all__ = ['ParametricModeling']