#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:14:19 2018

@author: arent
"""

import sys
sys.path.append('../classes')

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from datetime import datetime

from Models import Model
from GenData import GenData
from Prior import Prior
from Objfn import ObjectiveFn

C_m_type = 'vector'
np.random.seed(100)

# parameters for Model
N = 32
h = 0.5
eps_0 = 5e-4
T_inf = 25

z = np.linspace(0, 1, N+1)

# parameters for GenData
M = 32

# parameters for Prior
beta_prior = np.ones((N-1,))
sigma_prior = 1
C_beta = np.diag(sigma_prior**2*np.ones((N-1,)))

# start
model = Model(N, eps_0, h, T_inf)
prior = Prior(beta_prior, C_beta)
data = GenData(M, h, T_inf)


R = np.linalg.cholesky(C_beta)

N_samples = int(1e3)

beta_samp_lst = []
T_samp_lst = []
for i in range(N_samples):
    s = np.random.randn(N-1)
    
    beta_sample = beta_prior+R.dot(s)
    T_sample, _ = model.get_T_mult(beta_sample)
    
    T_samp_lst.append(T_sample)
    
T_data = np.vstack(T_samp_lst)
std_T = np.std(T_data, axis=0)

T_true, _ = data.get_T_true(with_rand=False)
T_prior_mean, _ = model.get_T_mult(beta_prior)

plt.figure()
plt.plot(z[1:-1], T_true, label='True')
plt.plot(z[1:-1], T_prior_mean, label='Prior')
plt.fill_between(z[1:-1], T_prior_mean-2*std_T, T_prior_mean+2*std_T, alpha=0.2)
plt.legend()
plt.show()





