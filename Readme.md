ParametricModeling
============


testfwefw
ParametricModeling is a Python class for non-linear as well as linear parametric
model estimation implemented as an sklearn interface around scipy.optimize.least_squares. 
It is distributed under the 3-Clause BSD license.

least_squares can be used to solve a nonlinear least-squares problem with bounds 
on the variables.
The aim of this project is to enclose scipy.optimize.least_squares 
function in a sklearn type interface for easier usage and ensure compatibility 
with sklearn tools like grid_search.

Important links
===============

- Official source code repo: http://github.com/shubhankar90/ParametricModeling
- Sklearn Development Guide: http://scikit-learn.org/stable/developers/index.html
- Scipy least_squares documentation: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

Dependencies
============
Created and tested to work under Python 3.4.4.1, scipy 0.17.0, sklearn 0.17.1, numpy 1.10.4.
All libraries mentioned above are required.

Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

Hosted on Pypi server. Use pip to install::

pip install ParametricModeling


Example Usage
-------------
Use "import ParametricModeling.ParametricModeling as PM" to import.
To check default argument values check docstring by "PM?".
If no parametric function is supplied a linear regression function is used by default.

  
	
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


	-----------------------------------------------------------------------------------------------------------

	############Data Generation from Linear function#############################    
	def gen_data(x, a, b, c, noise=0, n_outliers=0, random_state=0):
		x=x.T
		y = (a * (x[0]) + (x[1])*(b) ) + (x[2]* c)
		rnd = np.random.RandomState(random_state)
		error = noise * rnd.randn(len(x[0]))
		outliers = rnd.randint(0, len(x[0]), n_outliers)
		error[outliers] *= 10
		return y + error

	a = 3
	b = 2
	c = -1
	n_points = 100

	x_train = np.array([[randrange(5,200) for x in range(0,n_points)],[randrange(400,800) for x in range(0,n_points)]
						 ,[randrange(800,1200) for x in range(0,n_points)]]).T

	y_train = gen_data(x_train, a, b, c, noise=0.1, n_outliers=3)

	############Data Generation from Linear Function#############################

	#Mon-residual linear function
	def func_1(para, x):
		return ((para[0] * x[0] + (para[1] * (x[1])) + (para[2] * x[2])) )
	x0_1 = np.array([1.0, 1.0, 0.0])

	#Biased non-residual linear function    
	def func_2(para, x):
		return(para[0] * x[0] + (para[1] * (x[1])) )

	x0_2 = np.array([1.0, 1.0])

	#Biased non-residual linear function with constant   
	def func_3(para, x):
		return(para[0] * x[0] + (para[1] * (x[1])) ) + para[2]

	x0_3 = np.array([1.0, 1.0, 4])
		
	#Biased residual linear function    
	def func_4(para, x, y):
		return(para[0] * x[0] + (para[1] * (x[1])) ) + para[2] - y

	x0_4 = np.array([1.0, 1.0, 4])



	test_1 = PM(x0=x0_1, user_para_function=func_1, is_residual_function=False)
	model = test_1.fit(x_train, y_train)
	print(test_1.score(x_train, y_train))

	test_2 = PM(x0=x0_2, user_para_function=func_2, is_residual_function=False)
	model = test_2.fit(x_train, y_train)
	print(test_2.score(x_train, y_train))

	# Using sklearn Grid search to choose between func1 and func2 parametric forms
	from sklearn.grid_search import GridSearchCV

	search_param = {'x0': [(2,3,5),(1,400,4)], 'user_para_function':[func_1,func_2]}
	GS=GridSearchCV(model, search_param, verbose=True)
	GS_fit=GS.fit(x_train, y_train)
	print(GS_fit.best_params_)    

	test_3 = PM(x0=x0_3, user_para_function=func_3, is_residual_function=False)
	model = test_3.fit(x_train, y_train)
	print(test_3.score(x_train, y_train))

	test_4 = PM(x0=x0_4, user_para_function=func_4, is_residual_function=True)
	model = test_4.fit(x_train, y_train)
	print(test_4.score(x_train, y_train))

	# Testing for default function definition.
	# When no parametric function is supplied by the user,
	# the function fits a linear parametric function to the data.
	# Result is same as test_1
	test_5 = PM()
	model = test_5.fit(x_train, y_train)
	print(test_5.score(x_train, y_train))


Testing
-------

Example usage and testing code is available in http://github.com/shubhankar90/ParametricModeling/tests/test_ParametricModeling.py
