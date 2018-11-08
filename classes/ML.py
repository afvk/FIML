#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:25:11 2018

@author: arent
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split


from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam


class Dataset:
    
    
    def __init__(self, path_train, path_test):
        self.path_train = path_train
        self.path_test = path_test
        
        self.x_train, self.y_train = self.load_data_train()
        self.x_test, self.y_test = self.load_data_test()
        
        self.x_train, self.x_val, self.y_train, self.y_val = self.split()
        
    
    def load_data_train(self):
        (T_inf_train, 
         T_obs_train, 
         beta_MAP_train, 
         std_lst_train) = pickle.load(open(self.path_train, 'rb'))

        T_inf_train = np.vstack(T_inf_train).flatten()
        T_obs_train = np.vstack(T_obs_train).flatten()
        beta_MAP_train = np.vstack(beta_MAP_train).flatten()
        
        x_train = np.concatenate([T_obs_train[:,np.newaxis], T_inf_train[:,np.newaxis]], 
                         axis=1)
        y_train = beta_MAP_train
        return x_train, y_train


    def load_data_test(self):
        (T_inf_test, 
         T_true_test, 
         beta_true_test) = pickle.load(open(self.path_test, 'rb'))
        
        T_inf_test = np.vstack(T_inf_test).flatten()
        T_true_test = np.vstack(T_true_test).flatten()
        beta_true_test = np.vstack(beta_true_test).flatten()

        x_test = np.concatenate([T_true_test[:,np.newaxis], 
                                 T_inf_test[:,np.newaxis]], axis=1)
        y_test = beta_true_test
        return x_test, y_test
    
    
    def split(self):
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train)
        return x_train, x_val, y_train, y_val



class Network:
    
    def __init__(self, data, N_epochs, lr, N_hidden, N_nodes):
        self.data = data
        self.N_epochs = N_epochs
        self.lr = lr
        self.N_hidden = N_hidden
        self.N_nodes = N_nodes
        
        self.model = self.build_model()
    
    
    def build_model(self):
        inputs = Input(shape=(2,))
        x = Dense(self.N_nodes, activation='relu')(inputs)
        for i in range(self.N_hidden-1):
            x = Dense(self.N_nodes, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        opt = SGD(lr=self.lr)
        model.compile(optimizer=opt, loss='mse')
        
        return model
    
    
    def train(self):
        self.model.fit(self.data.x_train, self.data.y_train, 
                       epochs=self.N_epochs,
                       verbose=0,
                       validation_data=(self.data.x_val, self.data.y_val)) 

        mse = self.model.history.history['loss']
        mse_val = self.model.history.history['val_loss']
        
        return mse, mse_val




#class GaussianProcess:
#    '''
#    kernel = Kernel class with method kernel_func
#    '''
#    
#    def __init__(self, sigma_y, kernel):
#        self.sigma_y = sigma_y
#        self.kernel = kernel
#        
#    
#    
#    def train(self, x_train, y_train):
#        '''
#        Computes the matrices K and K_y for the Gaussian process.
#        '''
#        N = len(x_train)
#        
#        self.K = self.compute_K(x_train, x_train)
#        self.K_y = self.K+self.sigma_y**2*np.eye(N)
#        self.invK_y = np.linalg.inv(self.K_y)
#        
#        self.x_train = x_train
#        self.y_train = y_train
#
#
#    def compute_K(self, x, xp):
#        '''
#        Returns a matrix of the kernel output of all possible combinations of
#        an observation in array x and an observation in array xp. 
#        
#        x has shape (N, D)
#        xp has shape (Np, D)
#        '''
#        N = x.shape[0]
#        Np = xp.shape[0]
#        
#        K = np.ones((N, Np))
#        
#        for i in range(N):
#            for j in range(Np):
#                K[i,j] = self.kernel.kernel_func(x[i,:], xp[j,:])
#        
#        return K
#
#
#    def predict(self, x_test):
#        K_s = self.compute_K(self.x_train, x_test)
#        K_ss = self.compute_K(x_test, x_test)
#        
#        mu_s = K_s.T.dot(self.invK_y).dot(self.y_train)
#        Sigma_s = K_ss-K_s.T.dot(self.invK_y).dot(K_s)
#        
#        return mu_s, Sigma_s
#    
#    
#class RBF1d:
#    
#    def __init__(self, sigma_f, l):
#        self.sigma_f = sigma_f
#        self.l = l
#    
#    
#    def kernel_func(self, x, xp):
#        return self.sigma_f**2*np.exp(-(x-xp)**2/(2*self.l**2)) # RBF1d
#
#
#
#class RBFnd:
#    
#    def __init__(self, h):
#        self.h = h
#        
#        
#    def kernel_func(self, x, xp):
#        return np.exp(-np.linalg.norm(x-xp)**2/self.h**2)
#
#










