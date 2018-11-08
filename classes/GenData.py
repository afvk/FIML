#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:51:48 2018

@author: arent
"""

import numpy as np
import pandas as pd

def eps(T, rand):
    '''
    Calculate the emissivity as a stochastic non-linear function of temperature
    for the true model.
    '''
    return (1+5*np.sin(3*np.pi*T/200)+np.exp(0.02*T)+rand)*1e-4


class GenData:
    """
    Generate model training data from the 'true' differential equation.

    Args:
        M: Number of cells
        h: Convection coefficient
        T_inf: Surroundings temperature

    Returns:
        True solutions/data and covariance matrix.
    """
    
    def __init__(self, M, h, T_inf):
        self.M = M
        self.h = h
        self.T_inf = T_inf
        
        self.z_obs = np.linspace(0, 1, M+1)
        self.dz_obs = self.z_obs[1]-self.z_obs[0]    
        
#        self.T_obs, _ = self.get_T_true()
        
        
    def get_T_true(self, th=1e-7, it_max=50000, with_rand=True):
        """ Solve the forward problem (for the true model). """     
        
        T = np.zeros(self.M+1)
        T[0] = 0
        T[-1] = 0
        
        L2_lst = []
        L2 = 10
        
        if with_rand:
            rand = np.random.randn(self.M-1)*0.1
        else:
            rand = 0
        
        it = 0
        while L2>th and it<it_max:
            T_old = T.copy()
            
            new = 0.5*(T[:self.M-1]+T[2:] \
                       -self.dz_obs**2*(eps(T[1:self.M], rand)*(T[1:self.M]**4-self.T_inf**4) \
                                    +self.h*(T[1:self.M]-self.T_inf)))
            T[1:self.M] = 0.25*new+0.75*T[1:self.M]
            
            L2 = np.linalg.norm(T-T_old)
            L2_lst.append(L2)
            
            it += 1
            
        if it==it_max:
            print('Iterations for get_T_true not converged: L2 = %e'%L2)
        
        return T[1:-1], L2_lst
    

    def gen_data(self, N_samples, C_m_type, scalar_noise=None):
        """ Generate dataset and covariance matrix. """

        T_lst = []
        
        samp_lst = []
        for i in range(N_samples):
            T_true, _ = self.get_T_true()
            samp_lst.append(T_true)
            
        T_lst += samp_lst
        
        self.T_data = np.vstack(T_lst)
        
        if C_m_type=='scalar':
            self.C_m = np.diag(scalar_noise**2*np.ones((self.M-1,)))
            
        elif C_m_type=='vector':
            vector_noise = np.std(self.T_data, axis=0)
            self.C_m = np.diag(vector_noise**2)
            
        elif C_m_type=='matrix':
            self.C_m = np.cov(self.T_data, rowvar=False)
            
        self.T_obs = np.mean(self.T_data, axis=0)







