#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 21:47:32 2018

@author: arent
"""


import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from Models import Model
from GenData import GenData

N = 30

h = 0.5
eps_0 = 5e-4

z = np.linspace(0, 1, N+1)

plt.figure()
for T_inf in np.linspace(5,50,5):
    data = GenData(N, h, T_inf)
    mod = Model(N, eps_0, h, T_inf)
    
    T_true, _ = data.get_T_true()
    T_base, _ = mod.get_T_base()
    
    plt.plot(z[1:N], T_true, marker='s', fillstyle='none', linestyle='none', 
             color='k', label='True' if T_inf==5 else '')
    plt.plot(z[1:N], T_base, 'r', label='Base' if T_inf==5 else '')

plt.xlabel('z')
plt.ylabel('T')
plt.axis([0, 1, 0, 60])
plt.legend()
plt.tight_layout()
plt.savefig('../results/fig1.pdf')
plt.show()


