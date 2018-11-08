#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:39:08 2018

@author: arent
"""

import sys
sys.path.append('classes/')

import numpy as np
import matplotlib.pyplot as plt
import pickle


T_inf_train, T_obs_train, beta_MAP_train, std_lst_train = pickle.load(open('../data/ML.p','rb'))

T_inf_train = np.vstack(T_inf_train).flatten()
T_obs_train = np.vstack(T_obs_train).flatten()
beta_MAP_train = np.vstack(beta_MAP_train).flatten()


T_inf_test, T_true_test, beta_true_test = pickle.load(open('../data/MLtest.p','rb'))

T_inf_test = np.vstack(T_inf_test).flatten()
T_true_test = np.vstack(T_true_test).flatten()
beta_true_test = np.vstack(beta_true_test).flatten()



plt.subplot(211)
plt.plot(T_obs_train, beta_MAP_train, '.k', markersize=3, label='Train')
plt.plot(T_true_test, beta_true_test, '.r', markersize=1, label='Test')
plt.xlabel(r'$T$')
plt.ylabel(r'$\beta_{MAP}$')
plt.legend()

plt.subplot(212)
plt.plot(T_inf_train, beta_MAP_train, '.k', markersize=3, label='Train')
plt.plot(T_inf_test, beta_true_test, '.r', markersize=1, label='Test')
plt.xlabel(r'$T_\infty$')
plt.ylabel(r'$\beta_{MAP}$')
plt.legend()

plt.tight_layout()
plt.show()










