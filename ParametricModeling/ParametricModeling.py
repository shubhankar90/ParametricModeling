# -*- coding: utf-8 -*-
"""
This project is a sklearn interface wrapper around scipy.optimize.least_squares.
"""
# Authors: Shubhankar Mitra (shubhankar90@gmail.com)
#
# Licence: BSD 3 clause

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._lsq.common import EPS
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

__all__ = ['ParametricModeling']

# =============================================================================
# Sklearn interface class
# =============================================================================

class ParametricModeling(BaseEstimator,RegressorMixin):
    """
    This project is a sklearn interface wrapper around scipy.optimize.least_squares.
    least_squares can be used to solve a nonlinear least-squares problem with bounds 
    on the variables.
    The aim of this project is to enclose scipy.optimize.least_squares 
    function in a sklearn type interface for easier usage and ensure compatibility 
    with sklearn tools like grid_search.

    For more information on least_squares check the scipy reference page.
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    
    Parameters
    ----------
    x0 : array_like with shape (n,) or float
        Initial guess on independent variables. If float, it will be treated
        as a 1-d array with one element.
        
    user_para_function : callable
        Function which computes the vector of residuals, with the signature
        ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
        respect to its first argument. The argument ``x`` passed to this
        function is an ndarray of shape (n,) (never a scalar, even for n=1).
        It must return a 1-d array_like of shape (m,) or a scalar.
        Eg: When setting is_residual_function = False:
            def func(coeff, X):
                return(coeff[0] + coeff[1] * np.exp(coeff[2] * X))
            When setting is_residual_function = True:
            def func(coeff, X, y):
                return((coeff[0] + coeff[1] * np.exp(coeff[2] * X))-y)
            Where coeff is an array of the parameter values of the model.
                    
    is_residual_function : Boolean value to indicate if user_para_function 
        supplied providesthe residual value or only the parametric function output.
        
    jac : {'2-point', '3-point', 'cs', callable}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix, where
        element (i, j) is the partial derivative of f[i] with respect to
        x[j]). The keywords select a finite difference scheme for numerical
        estimation. The scheme '3-point' is more accurate, but requires
        twice as much operations compared to '2-point' (default). The
        scheme 'cs' uses complex steps, and while potentially the most
        accurate, it is applicable only when `fun` correctly handles
        complex inputs and can be analytically continued to the complex
        plane. Method 'lm' always uses the '2-point' scheme. If callable,
        it is used as ``jac(x, *args, **kwargs)`` and should return a
        good approximation (or the exact value) for the Jacobian as an
        array_like (np.atleast_2d is applied), a sparse matrix or a
        `scipy.sparse.linalg.LinearOperator`.
        
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each array must match the size of `x0` or be a scalar, in the latter
        case a bound will be the same for all variables. Use ``np.inf`` with
        an appropriate sign to disable bounds on all or some variables.
        
    method : {'trf', 'dogbox', 'lm'}, optional
        Algorithm to perform minimization.

            * 'trf' : Trust Region Reflective algorithm, particularly suitable
              for large sparse problems with bounds. Generally robust method.
            * 'dogbox' : dogleg algorithm with rectangular trust regions,
              typical use case is small problems with bounds. Not recommended
              for problems with rank-deficient Jacobian.
            * 'lm' : Levenberg-Marquardt algorithm as implemented in MINPACK.
              Doesn't handle bounds and sparse Jacobians. Usually the most
              efficient method for small unconstrained problems.

        Default is 'trf'. See Notes for more information.
    ftol : float, optional
        Tolerance for termination by the change of the cost function.
        Default is the square root of machine epsilon. The optimization process
        is stopped when ``dF < ftol * F``, and there was an adequate agreement
        between a local quadratic model and the true model in the last step.
    xtol : float, optional
        Tolerance for termination by the change of the independent variables.
        Default is the square root of machine epsilon. The exact condition
        checked depends on the `method` used:

            * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``
            * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
              a trust-region radius and ``xs`` is the value of ``x``
              scaled according to `x_scale` parameter (see below).

    gtol : float, optional
        Tolerance for termination by the norm of the gradient. Default is
        the square root of machine epsilon. The exact condition depends
        on a `method` used:

            * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
              ``g_scaled`` is the value of the gradient scaled to account for
              the presence of the bounds [STIR]_.
            * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
              ``g_free`` is the gradient with respect to the variables which
              are not in the optimal state on the boundary.
            * For 'lm' : the maximum absolute value of the cosine of angles
              between columns of the Jacobian and the residual vector is less
              than `gtol`, or the residual vector is zero.

    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting `x_scale` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        An alternative view is that the size of a trust-region along j-th
        dimension is proportional to ``x_scale[j]``. Improved convergence may
        be achieved by setting `x_scale` such that a step of a given length
        along any of the scaled variables has a similar effect on the cost
        function. If set to 'jac', the scale is iteratively updated using the
        inverse norms of the columns of the Jacobian matrix (as described in
        [JJMore]_).
    loss : str or callable, optional
        Determines the loss function. The following keyword values are allowed:

            * 'linear' (default) : ``rho(z) = z``. Gives a standard
              least-squares problem.
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
              approximation of l1 (absolute value) loss. Usually a good
              choice for robust least squares.
            * 'huber' : ``rho(z) = z if z <= 1 else z**0.5 - 1``. Works
              similarly to 'soft_l1'.
            * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
              influence, but may cause difficulties in optimization process.
            * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
              a single residual, has properties similar to 'cauchy'.

        If callable, it must take a 1-d ndarray ``z=f**2`` and return an
        array_like with shape (3, m) where row 0 contains function values,
        row 1 contains first derivatives and row 2 contains second
        derivatives. Method 'lm' supports only 'linear' loss.
    f_scale : float, optional
        Value of soft margin between inlier and outlier residuals, default
        is 1.0. The loss function is evaluated as follows
        ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,
        and ``rho`` is determined by `loss` parameter. This parameter has
        no effect with ``loss='linear'``, but for other `loss` values it is
        of crucial importance.
    max_nfev : None or int, optional
        Maximum number of function evaluations before the termination.
        If None (default), the value is chosen automatically:

            * For 'trf' and 'dogbox' : 100 * n.
            * For 'lm' :  100 * n if `jac` is callable and 100 * n * (n + 1)
              otherwise (because 'lm' counts function calls in Jacobian
              estimation).

    diff_step : None or array_like, optional
        Determines the relative step size for the finite difference
        approximation of the Jacobian. The actual step is computed as
        ``x * diff_step``. If None (default), then `diff_step` is taken to be
        a conventional "optimal" power of machine epsilon for the finite
        difference scheme used [NR]_.
    tr_solver : {None, 'exact', 'lsmr'}, optional
        Method for solving trust-region subproblems, relevant only for 'trf'
        and 'dogbox' methods.

            * 'exact' is suitable for not very large problems with dense
              Jacobian matrices. The computational complexity per iteration is
              comparable to a singular value decomposition of the Jacobian
              matrix.
            * 'lsmr' is suitable for problems with sparse and large Jacobian
              matrices. It uses the iterative procedure
              `scipy.sparse.linalg.lsmr` for finding a solution of a linear
              least-squares problem and only requires matrix-vector product
              evaluations.

        If None (default) the solver is chosen based on type of Jacobian
        returned on the first iteration.
    tr_options : dict, optional
        Keyword options passed to trust-region solver.

            * ``tr_solver='exact'``: `tr_options` are ignored.
            * ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.
              Additionally  ``method='trf'`` supports  'regularize' option
              (bool, default is True) which adds a regularization term to the
              normal equations, which improves convergence if Jacobian is
              rank-deficient [Byrd]_ (eq. 3.4).

    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines the sparsity structure of the Jacobian matrix for finite
        differences. If the Jacobian has only few non-zeros in *each* row,
        providing the sparsity structure will greatly speed up the computations
        [Curtis]_. Should have shape (m, n). A zero entry means that a
        corresponding element in the Jacobian is identically zero. If provided,
        forces the use of 'lsmr' trust-region solver. If None (default) then
        dense differencing will be used. Has no effect for 'lm' method.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations (not supported by 'lm'
              method).

    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun` and `jac`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)`` and the same for
        `jac`.

    Attributes
    ----------
    coeff_ : Parameter values obtained through scipy.optimize.least_squares.
    
    residuals_ : Residual values.
    
    ls_out_ : scipy.optimize.least_squares result dictionary object.
    
    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Nonlinear_regression

    .. [2] http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.
         
    Examples
    --------
    ############Data Generation from exponential Function#############################
    import ParametricModeling.ParametricModeling as PM
    import numpy as np
    from random import randrange
    
    def gen_data(x, a, b, noise=0, n_outliers=0, random_state=0):
        x=x.T
        y = (a * (x[0])) + (np.log(b * x[1]))
        rnd = np.random.RandomState(random_state)
        error = noise * rnd.randn(len(x[0]))
        outliers = rnd.randint(0, len(x[0]), n_outliers)
        error[outliers] *= 10
        return y + error
    
    a = 3
    b = 2
    n_points = 100
    
    x_train = np.array([[randrange(5,200) for x in range(0,n_points)],[randrange(400,800) for x in range(0,n_points)]]).T
    
    y_train = gen_data(x_train, a, b, noise=0.1, n_outliers=3)
    
    ############Data Generation from exponential Function#############################
    
    #exponential residual function
    def func_6(para, x, y):
        return (para[0]*x[0]) + np.log(para[1] * x[1]) - y

    x0_6 = np.array([1.0, 1.0])
    
    test_6 = PM(x0=x0_6, user_para_function=func_6, is_residual_function=True)
    model = test_6.fit(x_train, y_train)
    print(test_6.score(x_train, y_train))
    """
    def __init__(self, x0=None, user_para_function=None, is_residual_function=True
        , jac='2-point',bounds=(-np.inf, np.inf), loss='linear', method='trf'
        , ftol=EPS**0.5, xtol=EPS**0.5, gtol=EPS**0.5, x_scale=1.0, f_scale=1.0
        , diff_step=None, tr_solver=None, tr_options=None, jac_sparsity=None
        , max_nfev=None, verbose=0, kwargs=None):
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
        return(np.array(self.user_para_function(para, x)) - y)

    def _default_para_func(self, para, x, y):
        #x=np.append(x,[[1 for x in range(np.shape(x)[1])]],axis=0)
        out = (np.dot(para,np.array(x))-y)
        return(out)

    def fit(self, X, y):
        """
        Find the best parameters for the user porvided parametric model using 
        scipy.optimise.least_squares.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
       
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        if self.x0 is None:
            self.x0 = [0 for i in range(np.shape(X)[1])]
        if self.kwargs is None:
            self.kwargs = {}
        if self.tr_options is None:
            self.tr_options = {}
        if self.user_para_function is None:
            func = self._default_para_func
            self.user_para_function = self._default_para_func
        elif self.is_residual_function == True:
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
        """
        Predict the value if y from the input array X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
       
        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        X = check_array(X)        
        if self.is_residual_function == True:
            return (self.user_para_function(self.coeff_, np.array(X).T, y=0))
        else:
            return (self.user_para_function(self.coeff_, np.array(X).T))
    
    def score(self, X, y):
        """
        Find the R square value from the predicted values using the fitted model
        and actual values.
        
        Parameters
        ----------
	
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
                 Ground truth (correct) target values.
        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
                Estimated target values.
       
        Returns
        -------
        C : float or ndarray of floats
            The R^2 score or ndarray of scores if ‘multioutput’ is ‘raw_values’.
        """
        X, y = check_X_y(X, y)
        residuals = self.predict(X) - y
        r_sq = 1 - (np.mean(residuals**2) /
        np.mean((y-np.mean(y))**2))
        return (r_sq)


