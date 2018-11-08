#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:51:45 2018

@author: arent
"""


import sys
sys.path.append('classes/')

import matplotlib.pyplot as plt
import numpy as np

from Models import Model
from Prior import Prior
from GenData import GenData
from Objfn import ObjectiveFn
from Optimization import Optimization


C_m_type = 'scalar'

N = 32
h = 0.5
eps_0 = 5e-4
T_inf = 50
z = np.linspace(0, 1, N+1)
M = 32
beta_prior = np.ones((N-1,))
sigma_prior = 0.8
C_beta = np.diag(sigma_prior**2*np.ones((N-1,)))


model = Model(N, eps_0, h, T_inf)
prior = Prior(beta_prior, C_beta)
data = GenData(M, h, T_inf)
data.gen_data(100, C_m_type=C_m_type, scalar_noise=0.02)

objfn = ObjectiveFn(z, data, model, prior)

beta0 = np.ones(N-1)

optimizers = ['BFGS','Newton-CG']
conv_lst = []

for optimizer in optimizers:
    opt = Optimization(model, prior, data, objfn, beta0, optimizer)
    conv_lst.append(opt.conv)


plt.figure()
for i, optimizer in enumerate(optimizers):
    plt.loglog(conv_lst[i], label=optimizer)
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('J')
plt.tight_layout()
plt.show()








