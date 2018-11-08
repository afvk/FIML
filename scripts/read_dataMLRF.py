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

from sklearn.ensemble import RandomForestRegressor
from matplotlib.font_manager import FontProperties

T_inf_train, T_obs_train, beta_MAP_train, std_list = pickle.load(open('../data/ML.p','rb'))

T_inf_train = np.vstack(T_inf_train).flatten()
T_obs_train = np.vstack(T_obs_train).flatten()
beta_MAP_train = np.vstack(beta_MAP_train).flatten()


T_inf_test, T_true_test, beta_true_test = pickle.load(open('../data/MLtest.p','rb'))

T_inf_test = np.vstack(T_inf_test).flatten()
T_true_test = np.vstack(T_true_test).flatten()
beta_true_test = np.vstack(beta_true_test).flatten()



x_train = np.concatenate([T_obs_train[:,np.newaxis], T_inf_train[:,np.newaxis]], 
                         axis=1)
y_train = beta_MAP_train

x_test = np.concatenate([T_true_test[:,np.newaxis], T_inf_test[:,np.newaxis]], 
                         axis=1)

RF = RandomForestRegressor()
RF.fit(x_train, y_train)
y_pred = RF.predict(x_test)


plt.figure()
plt.plot(y_pred, beta_true_test, '.')
plt.show()


plt.figure()
plt.subplot(211)
plt.plot(T_true_test, y_pred, '.k', label='Prediction')
plt.plot(T_true_test, beta_true_test, '.r', markersize=1, label='Test')
plt.xlabel(r'$T$')
plt.ylabel(r'$\beta$')
plt.legend()

plt.subplot(212)
plt.plot(T_inf_test, y_pred, '.k', label='Prediction')
plt.plot(T_inf_test, beta_true_test, '.r', markersize=1, label='Test')
plt.xlabel(r'$T_\infty$')
plt.ylabel(r'$\beta$')
plt.legend()
plt.tight_layout()
plt.savefig('../results/T_prediction_RF.eps')
plt.show()

#
#N = 32
#z = np.linspace(0, 1, N+1)
#eps_0 = 5e-4
#h = 0.5
#
#
#T_inf = dict(func=28*np.ones_like(z[1:-1]),
#             string=r'$T_\infty = 28$')
#T_inf = dict(func=55*np.ones_like(z[1:-1]),
#             string=r'$T_\infty = 55$')
#T_inf = dict(func=35+20*np.sin(2*np.pi*z[1:-1]),
#             string=r'$T_\infty = 35+20*\sin(2\pi z)$')
#T_inf = dict(func=35-15*z[1:-1],
#             string=r'$T_\infty = 35-15z$')
#T_inf = dict(func=15+5*np.cos(np.pi*z[1:-1]),
#             string=r'$T_\infty = 15+5 \cos(\pi z)$')
#
#T_inf_lst = [
#             dict(func=28*np.ones_like(z[1:-1]), string=r'$T_\infty = 28$'),
#             dict(func=55*np.ones_like(z[1:-1]), string=r'$T_\infty = 55$'),
#             dict(func=35+20*np.sin(2*np.pi*z[1:-1]), string=r'$T_\infty = 35+20*\sin(2\pi z)$'),
#             dict(func=35-15*z[1:-1], string=r'$T_\infty = 35-15z$'),
#             dict(func=15+5*np.cos(np.pi*z[1:-1]),  string=r'$T_\infty = 15+5 \cos(\pi z)$')
#            ]
#
#
#fontP = FontProperties()
#fontP.set_size('small')
#
#plt.figure()
#for i,T_inf in enumerate(T_inf_lst):
#    model = Model(N, eps_0, h, T_inf['func'])
#    T, conv = model.get_T_ML(RF, th=1e-4, it_max=50000)
#    
#    data = GenData(N, h, T_inf['func'])
#    T_true, _ = data.get_T_true()
#
#    plt.plot(z[1:-1], T_true, marker='s', fillstyle='none', linestyle='none', 
#                 color='k', label='True' if i==0 else '')
#    plt.plot(z[1:-1], T, label=T_inf['string'])
#plt.xlabel(r'$z$')
#plt.ylabel(r'$T$')
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.tight_layout()
#plt.savefig('../results/ML_solutions_RF.eps')
#plt.show()


#plt.figure()
#plt.semilogy(conv)
#plt.xlabel('Iterations')
#plt.ylabel('L2 Residual')
#plt.show()















