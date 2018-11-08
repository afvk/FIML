#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:32:21 2018

@author: arent
"""

import sys
sys.path.append('classes/')

from ML import GaussianProcess, RBFnd
from Models import Model
from GenData import GenData

import numpy as np
import matplotlib.pyplot as plt
import pickle


T_inf_train, T_obs_train, beta_MAP_train, std_lst = pickle.load(open('data/ML.p','rb'))

T_inf_train = np.vstack(T_inf_train).flatten()
T_obs_train = np.vstack(T_obs_train).flatten()
beta_MAP_train = np.vstack(beta_MAP_train).flatten()
std_lst = np.vstack(std_lst).flatten()


T_inf_test, T_true_test, beta_true_test = pickle.load(open('data/MLtest.p','rb'))

T_inf_test = np.vstack(T_inf_test).flatten()
T_true_test = np.vstack(T_true_test).flatten()
beta_true_test = np.vstack(beta_true_test).flatten()

#plt.figure()
#plt.plot(T_obs_train, beta_MAP_train, '.k', label='Train')
#plt.plot(T_true_test, beta_true_test, '.r', label='Test')
#plt.legend()
#plt.show()

x_train = np.concatenate([T_obs_train[:,np.newaxis],T_inf_train[:,np.newaxis]], 
                         axis=1)
y_train = beta_MAP_train[:,np.newaxis]

h = 20
kernel = RBFnd(h)

sigma_y = 0.1
GP = GaussianProcess(sigma_y, kernel)
GP.train(x_train, y_train)

x_test = np.concatenate([T_true_test[:,np.newaxis],T_inf_test[:,np.newaxis]], 
                         axis=1)

mupost, sigmapost = GP.predict(x_test)
std = np.sqrt(np.diag(sigmapost))

plt.figure()
plt.plot(T_obs_train, beta_MAP_train, '.k', label='Train')
plt.plot(T_true_test, beta_true_test, '.r', label='Test')
#plt.fill_between(x_test[:,0], mupost[:,0]-2*std, mupost[:,0]+2*std, alpha=0.2)
plt.plot(x_test[:,0], mupost, '.b', label='Post')
plt.legend()
plt.show()


plt.figure()
plt.plot(T_inf_train, beta_MAP_train, '.k', label='Train')
plt.plot(T_inf_test, beta_true_test, '.r', label='Test')
#plt.fill_between(x_test[:,0], mupost[:,0]-2*std, mupost[:,0]+2*std, alpha=0.2)
plt.plot(x_test[:,1], mupost, '.b', label='Post')
plt.legend()
plt.show()



#N = 32
#z = np.linspace(0, 1, N+1)
#eps_0 = 5e-4
#T_inf = "15+5*np.cos(np.pi*z[1:-1])"
##T_inf = 35-15*z[1:-1]
#
#model = Model(N, eps_0, h, eval(T_inf))
#T, conv = model.get_T_ML(GP, th=1e-4, it_max=50000)
#
#
#data = GenData(N, h, eval(T_inf))
#T_true, _ = data.get_T_true()
#
#plt.figure()
#plt.plot(z[1:-1], T, label='ML')
#plt.plot(z[1:-1], T_true, label='True')
#plt.title(T_inf)
#plt.legend()
#plt.show()




















