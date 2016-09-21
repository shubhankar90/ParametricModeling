"""
Testing Parametric Modeling
"""

# Authors: Shubhankar Mitra <shubhankar90@gmail.com>
#          
# License: BSD 3 clause

import ParametricModeling.ParametricModeling as PM

import numpy as np
from random import randrange

if __name__ == "__main__":
    
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
    

############Data Generation from exponential function#############################    
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