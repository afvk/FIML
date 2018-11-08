#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:35:04 2018

@author: arent
"""


import numpy as np

class Model:
    """
    Solvers for the approximate model, augmented model, and the 
    machine learning model. 

    Args:
        M: Number of cells
        eps_0: Emissivity approximation
        h: Convection coefficient
        T_inf: Surroundings temperature

    Returns:
        Modeled solutions.
    """
    
    def __init__(self, N, eps_0, h, T_inf):
        self.N = N
        self.eps_0 = eps_0
        self.h = h
        self.T_inf = T_inf
        
        self.z = np.linspace(0, 1, N+1)
        self.dz = self.z[1]-self.z[0]
        
    
    def get_beta_r_true(self, T, with_rand=True):
        if with_rand:
            rand = np.random.normal(loc=0, scale=0.1, size=(T.shape))
        else:
            rand = 0
        return (1/self.eps_0)*(1+5*np.sin(3*np.pi*T/200)+np.exp(0.02*T)+rand)*1e-4
        
        
    def get_beta_c_true(self, T):
        return self.h/self.eps_0*(self.T_inf-T)/(self.T_inf**4-T**4)
    
    
    def get_beta_true(self, T, with_rand=True):
        """ 
        Calculate the true discrepancy function given the true model and the base
        model.
        """
        
        beta_r = self.get_beta_r_true(T, with_rand)
        beta_c = self.get_beta_c_true(T)
        return beta_r+beta_c
    
    
    def get_T_base(self, th=1e-7, it_max=50000):
        """ Solve the forward problem for the base model."""
        
        T = np.zeros(self.N+1)
        T[0] = 0
        T[-1] = 0
        
        L2_lst = []
        L2 = 10
        
        it = 0    
        while L2>th and it<it_max:
            T_old = T.copy()
    
            new = 0.5*(T[:self.N-1]+T[2:]-self.dz**2*self.eps_0*(T[1:self.N]**4-self.T_inf**4))
            T[1:self.N] = 0.25*new+0.75*T[1:self.N]
            
            L2 = np.linalg.norm(T-T_old)
            L2_lst.append(L2)
            
            it += 1
        
        if it==it_max:
            print('Iterations for get_T_base not converged: L2 = %e'%L2)
    
        return T[1:-1], L2_lst
    
    
    def get_T_mult(self, beta, th=1e-7, it_max=50000):
        '''
        Solve the forward problem for the multiplied model.
        '''
        T = np.zeros(self.N+1)
        T[0] = 0
        T[-1] = 0
        
        L2_lst = []
        L2 = 10
        
        it = 0    
        while L2>th and it<it_max:
            T_old = T.copy()
    
            new = 0.5*(T[:self.N-1]+T[2:]- \
                       self.dz**2*self.eps_0*beta*(T[1:self.N]**4-self.T_inf**4))
            T[1:self.N] = 0.25*new+0.75*T[1:self.N]
            
            L2 = np.linalg.norm(T-T_old)
            L2_lst.append(L2)
            
            it += 1
        
        if it==it_max:
            print('Iterations for get_T_base not converged: L2 = %e'%L2)
    
        return T[1:-1], L2_lst
    
    
    def get_T_ML(self, ML_model, th=1e-7, it_max=50000):
        
        T = np.zeros(self.N+1)
        T[0] = 0
        T[-1] = 0
        
        L2_lst = []
        L2 = 10
        
        it = 0    
        while L2>th and it<it_max:
            if it%100==0:
                print(it, L2)
            T_old = T.copy()
            
            beta = np.ones((self.N-1))
            for i in range(1,self.N):
                beta[i-1] = ML_model.predict(np.array([T[i], self.T_inf[i-1]]).reshape(1,-1))[0]
                
            
            new = 0.5*(T[:self.N-1]+T[2:]- \
                       self.dz**2*self.eps_0*beta*(T[1:self.N]**4-self.T_inf**4))
            
            alpha = 0.5
            T[1:self.N] = alpha*new+(1-alpha)*T[1:self.N]
            
            L2 = np.linalg.norm(T-T_old)
            L2_lst.append(L2)
            
            it += 1
        
        if it==it_max:
            print('Iterations for get_T_ML not converged: L2 = %e'%L2)
    
        return T[1:-1], L2_lst
    















