#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:40:33 2018

@author: arent
"""

import sys
sys.path.append('classes/')

import numpy as np

from Models import Model
from GenData import GenData
from Prior import Prior
from Objfn import ObjectiveFn
from Optimization import Optimization

import pickle

C_m_type = 'matrix'
np.random.seed(100)

N = 32
h = 0.5
eps_0 = 5e-4
z = np.linspace(0, 1, N+1)
M = 32
beta_prior = np.ones((N-1,))

T_inf_lst = np.arange(5, 51, 5)
sigma_prior_lst = [20, 2, 1, 1, 0.5, 1, 1, 1 ,1, 0.8]
T_MAP_lst = []
beta_r_MAP_lst = []
beta_c_MAP_lst = []
std_lst = []

for i,T_inf in enumerate(range(5, 51, 5)):
    sigma_prior = sigma_prior_lst[i]
    C_beta = np.diag(sigma_prior**2*np.ones((N-1,)))
    
    model = Model(N, eps_0, h, T_inf)
    prior = Prior(beta_prior, C_beta)
    data = GenData(M, h, T_inf)
    data.gen_data(100, C_m_type=C_m_type, scalar_noise=0.02)
    
    objfn = ObjectiveFn(z, data, model, prior)
    
    beta0 = np.ones((N-1,))
    optimizer = 'Newton-CG' 
    
    opt = Optimization(model, prior, data, objfn, beta0, optimizer)
    
    opt.compute_MAP_properties()
    opt.compute_true_properties()
    opt.compute_base_properties()
    opt.sample_posterior(int(1e6))
    
    T_MAP_lst.append(opt.T_MAP)
    beta_r_MAP_lst.append(opt.beta_r_MAP)
    beta_c_MAP_lst.append(opt.beta_c_MAP)
    std_lst.append(opt.std)
    
pickle.dump((T_MAP_lst, 
             beta_r_MAP_lst, 
             beta_c_MAP_lst, 
             std_lst, 
             C_m_type), open('data/fig3%s.p'%C_m_type,'wb'))

