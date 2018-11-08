#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 16:46:02 2018

@author: arent
"""

#
#def solve_inverse_problem(objfn, beta0):
#    convJ = [objfn.compute_J(beta0)]
#    
#    N_it = 5000
#    lr = 0.00001
#    
#    beta = beta0.copy()
#    for i in range(N_it):
#        grad = objfn.compute_gradient_adjoint(beta)
#        beta -= grad*lr
#        
#        convJ += [objfn.compute_J(beta)]
#
#    return beta, convJ

def steepest_descent(fun, x0, jac, *args):
    x = x0.copy()
    conv = []
    
    for i in range(niter):
        x -= jac(x)*lr
        
        conv += [fun(x)]

    return x, conv

