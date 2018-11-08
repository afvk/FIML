#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:53:38 2018

@author: arent
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

from GenData import GenData
from Models import Model

np.random.seed(100)

C_m_type = 'vector'
N = 32
h = 0.5
eps_0 = 5e-4
z = np.linspace(0, 1, N+1)
M = 32


beta_prior = np.ones((N-1,))
sigma_prior = 0.8
C_beta = np.diag(sigma_prior**2*np.ones((N-1,)))

(T_MAP_lst, 
 beta_r_MAP_lst, 
 beta_c_MAP_lst, 
 std_lst, C_m_type) = pickle.load(open('../data/fig3%s.p'%C_m_type,'rb'))


f = plt.figure(figsize=(14,4))

ax1 = f.add_subplot(131)
ax2 = f.add_subplot(132)
ax3 = f.add_subplot(133)

for i,T_inf in enumerate(range(5, 51, 5)):
    data = GenData(M, h, T_inf)
    T_true, _ = data.get_T_true(with_rand=False)
    model = Model(N, eps_0, h, T_inf)
    beta_r_true = model.get_beta_r_true(T_true, with_rand=False)
    ax1.plot(T_true, beta_r_true, 'k', label='True' if i ==0 else '')

    T_MAP = T_MAP_lst[i]
    beta_r_MAP = beta_r_MAP_lst[i]
    std = std_lst[i]

    ax1.errorbar(T_MAP, beta_r_MAP, yerr=std, marker='o', fillstyle='none', 
             linestyle='none', mec='r', ecolor='0.5', ms=4, capsize=3, 
             label='MAP' if i==0 else '')

ax1.set_xlabel(r'$T$')
ax1.set_ylabel(r'$\beta_r$')
if C_m_type=='matrix':
    ax1.axis([0, 50, 0.2, 1.8])
elif C_m_type=='vector':
    ax1.axis([0, 50, -1, 2.5])
elif C_m_type=='scalar':
    ax1.axis([0, 50, -0.5, 2.5])

ax1.legend()

for i,T_inf in enumerate(range(5, 51, 5)):
    data = GenData(M, h, T_inf)
    T_true, _ = data.get_T_true()
    model = Model(N, eps_0, h, T_inf)
    beta_c_true = model.get_beta_c_true(T_true)
    ax2.semilogy(T_true, beta_c_true, 'k', label='True' if i==0 else '')
    
    T_MAP = T_MAP_lst[i]
    beta_c_MAP = beta_c_MAP_lst[i]
    
    ax2.plot(T_MAP, beta_c_MAP, marker='o', fillstyle='none', 
             linestyle='none', color='r', ms=4, label='MAP' if i==0 else  '')

ax2.set_xlabel(r'$T$')
ax2.set_ylabel(r'$\beta_c$')
ax2.axis([0, 50, 1e-3, 1e1])
ax2.legend()


for i,T_inf in enumerate(range(5, 51, 5)):
    T_MAP = T_MAP_lst[i]
    std = std_lst[i]
    
    ax3.plot(T_MAP, std, marker='o', fillstyle='none', 
             linestyle='none', color='r', ms=4, label='MAP' if i==0 else '')

ax3.axhline(0.02, color='k', label='True')
ax3.set_xlabel(r'$T$')
ax3.set_ylabel(r'$\sigma$')
if C_m_type=='scalar' or C_m_type=='vector':
    ax3.axis([0, 50, 0, 1])
elif C_m_type=='matrix':
    ax3.axis([0, 50, 0.0185, 0.0225])

ax3.legend()
f.tight_layout()
f.savefig('../results/fig3_%s.eps'%C_m_type)
f.show()










