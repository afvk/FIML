#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:20:46 2018

@author: arent
"""


import copy

import numpy as np
import scipy.optimize as optimize



class Optimization:
    """
    Optimization and post-processing of field inversion.

    Args:
        model: Approximate model 
        prior: Prior distribution
        data: Observations of the physical process
        objfn: Objective function
        beta0: Initial corrective term
        optimizer: Optimization algorithm

    Returns:
        Optimized corrective term.
    """
    
    def __init__(self, model, prior, data, objfn, beta0, optimizer):
        self.model = model
        self.prior = prior
        self.data = data
        self.objfn = objfn
        
        self.solve_inverse_problem(self.objfn, beta0, optimizer)
        
        
    def J(self, beta):
        tmp = self.objfn.compute_J(beta)

        self.convI += [tmp]
        return tmp


    def solve_inverse_problem(self, objfn, beta0, optimizer):
        
        beta = copy.deepcopy(beta0)
        
        self.convI = []
    
        res = optimize.minimize(self.J, beta, method=optimizer, 
                                jac=objfn.compute_gradient_adjoint,
                                options={'disp': True, 'maxiter': 1000})

        self.beta_MAP = res.x
        self.conv = np.array(self.convI)
        
    
    def compute_MAP_properties(self):
        self.T_MAP, _ = self.model.get_T_mult(self.beta_MAP)
        
        self.beta_r_MAP = self.model.get_beta_r_true(self.T_MAP)
        self.beta_c_MAP = self.model.get_beta_c_true(self.T_MAP)
        
        
    def compute_true_properties(self):
        self.T_true, _ = self.data.get_T_true()
        self.beta_true = self.model.get_beta_true(self.T_true, with_rand=False)
        self.J_limit = self.objfn.compute_J(self.beta_true)
        
    
    def compute_base_properties(self):
        self.T_base, _ = self.model.get_T_base()

        
    def sample_posterior(self, N_samples):
        H = self.objfn.compute_Hessian_adjoint_direct(self.beta_MAP)
        C_MAP = np.linalg.inv(H)
        R = np.linalg.cholesky(C_MAP)
        
        samp_lst = []
        for i in range(N_samples):
            s = np.random.randn(self.model.N-1)
            samp_lst.append(self.beta_MAP+R.dot(s))
        
        samp = np.vstack(samp_lst)
        self.std = np.std(samp, axis=0)















