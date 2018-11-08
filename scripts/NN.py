#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:48:00 2018

@author: arent
"""


import sys
sys.path.append('../classes')

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


from ML import Dataset, Network

np.random.seed(123)

path_train = '../data/ML.p'
path_test = '../data/MLtest.p'

data = Dataset(path_train, path_test)

N_epochs = 10000
lr_lst = [1e-2, 1e-3, 1e-4]
N_hidden_lst = [0, 1, 2]
N_nodes_lst = [8, 16]


results = {}

for lr in tqdm(lr_lst):
    results[lr] = dict()
    for N_hidden in tqdm(N_hidden_lst):
        results[lr][N_hidden] = dict()
        for N_nodes in tqdm(N_nodes_lst):
            results[lr][N_hidden][N_nodes] = dict()
            
            NN = Network(data, N_epochs, lr, N_hidden, N_nodes)
            mse, mse_val = NN.train()
            
            results[lr][N_hidden][N_nodes]['mse'] = mse
            results[lr][N_hidden][N_nodes]['mse_val'] = mse_val
            
            


plt.figure()

for lr in lr_lst:
    for N_hidden in N_hidden_lst:
        for N_nodes in N_nodes_lst:
            
            
            mse = results[lr][N_hidden][N_nodes]['mse']
            mse_val = results[lr][N_hidden][N_nodes]['mse_val']
            
            plt.semilogy(mse_val, label='lr = %.1e, N_hidden = %i, N_nodes = %i'%(lr, N_hidden, N_nodes))

plt.legend()

plt.show()











#plt.figure()
#plt.plot(data.y_val, model.predict(data.x_val), '.k')
#plt.plot([data.y_val.min(), data.y_val.max()],
#         [data.y_val.min(), data.y_val.max()], 'k')
#plt.axis('equal')
#plt.show()


#plt.figure()
#plt.plot(data.y_test, model.predict(data.x_test), '.k')
#plt.plot([data.y_test.min(), data.y_test.max()],
#         [data.y_test.min(), data.y_test.max()], 'k')
#plt.axis('equal')
#plt.show()


#plt.figure()
#plt.semilogy(model.history.history['loss'], label='loss')
#plt.semilogy(model.history.history['val_loss'], label='val_loss')
#plt.show()








