# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:05:57 2016

@author: Shubhankar Mitra
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._lsq.common import EPS
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array



class ParametricModeling(BaseEstimator):
    
    def __init__(self, x0=None, user_para_function=None, is_residual_function=True
                , jac='2-point',bounds=(-np.inf, np.inf), loss='linear', method='trf'
                , ftol=EPS**0.5, xtol=EPS**0.5, gtol=EPS**0.5, x_scale=1.0, f_scale=1.0
                , diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None
                , max_nfev=None, verbose=0, kwargs={}):
        self.x0 = x0
        self.is_residual_function = is_residual_function
        self.user_para_function = user_para_function
        self.jac = jac
        self.bounds = bounds
        self.loss = loss
        self.method = method
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.x_scale = x_scale
        self.f_scale = f_scale
        self.diff_step = diff_step
        self.tr_solver = tr_solver
        self.tr_options = tr_options
        self.jac_sparsity = jac_sparsity
        self.max_nfev = max_nfev
        self.verbose = verbose
        self.kwargs = kwargs
        
    def _para_residual_function(self, para, x, y):
        return np.array(self.user_para_function(para, x)) - y
    
    def _default_para_func(x, t, y):
        return(1)
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.x0 is None:
            self.x0 = [0 for i in range(np.shape(X)[2])]
        if self.user_para_function is None:
            self.user_para_function = self._default_para_func
        if self.is_residual_function == True:
            func = self.user_para_function
        else:
            func = self._para_residual_function
        res = least_squares(func, self.x0, jac=self.jac, bounds=self.bounds, method=self.method,
            ftol=self.ftol, xtol=self.xtol, gtol=self.gtol, x_scale=self.x_scale,
            loss=self.loss, f_scale=self.f_scale, diff_step=self.diff_step, tr_solver=self.tr_solver,
            tr_options=self.tr_options, jac_sparsity=self.jac_sparsity, max_nfev=self.max_nfev
            , verbose=self.verbose, args=(X.T,y),
            kwargs=self.kwargs)
        self.residuals_ = res.fun
        self.coeff_ = res.x
        self.ls_out_ = res
        return self
    
    def predict(self, X):
        X = check_array(X)        
        if self.is_residual_function == True:
            return self.user_para_function(self.coeff_, np.array(X).T, y=0)
        else:
            return self._para_residual_function(self.coeff_, np.array(X).T)
 
    def score(self, X, y):
        X, y = check_X_y(X, y)
        residuals = self.predict(X) - y
        r_sq = 1 - (np.mean(residuals**2) /
        np.mean((y-np.mean(y))**2))
        return r_sq
    
