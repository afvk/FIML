#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:41:55 2018

@author: arent
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('classes/')

from Models import Model
from GenData import GenData
from Prior import Prior
from Objfn import ObjectiveFn

np.random.seed(50)

N = 40
h = 0.5
eps_0 = 5e-4
T_inf = 50

z = np.linspace(0, 1, N+1)
M = 20

beta_prior = np.ones((N-1,))
sigma_prior = 1e-1
C_beta = np.diag(sigma_prior**2*np.ones((N-1,)))


model = Model(N, eps_0, h, T_inf)
prior = Prior(beta_prior, C_beta)
data = GenData(M, h, T_inf)
data.gen_data(2, 'vector', scalar_noise=0.02)
objfn = ObjectiveFn(z, data, model, prior)

data_true = GenData(N, h, T_inf)
T_true, _ = data_true.get_T_true()
beta_test = model.get_beta_true(T_true)

G_adjoint = objfn.compute_gradient_adjoint(beta_test)
G_direct = objfn.compute_gradient_direct(beta_test)
G_adjoint_cont = objfn.compute_gradient_adjoint_cont(beta_test)


plt.figure()
for epsilon in [-1,-2,-3,-4]:
    G_findiff = objfn.compute_gradient_findiff(beta_test, epsilon=10**epsilon)
    plt.plot(z[1:-1], G_findiff, label='FD, eps = '+str(epsilon))
plt.plot(z[1:-1], G_direct, '.r', label='Direct')
plt.plot(z[1:-1], G_adjoint, 'xk', label='Discrete adjoint')
plt.plot(z[1:-1], G_adjoint_cont, marker='s', fillstyle='none', linestyle='none', 
             color='k', label='Continuous adjoint')
plt.xlabel(r'$z$')
plt.ylabel(r'$\delta J/ \delta \beta_n$')
plt.legend()
plt.tight_layout()
plt.savefig('../results/gradient_verification.pdf')
plt.show()
    


