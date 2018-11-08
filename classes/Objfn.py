#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:21:53 2018

@author: arent
"""


import numpy as np
from numdifftools.core import Hessian
import matplotlib.pyplot as plt

class ObjectiveFn:
    """
    Definition of the objective function and all variants of direct/adjoint
    methods to obtain the Jacobian and the Hessian to be used in the 
    optimization. 

    Args:
        z: Discretized domain
        data: Observations of the physical process.
        model: Approximate model 
        prior: Prior distribution

    Returns:
        Modeled solutions.
    """
    
    def __init__(self, z, data, model, prior):
        self.N = len(z)-1
        self.z = z
        self.dz = z[1]-z[0]
        self.model = model
        self.data = data
        self.prior = prior
        
        self.H_matrix = self.compute_H_matrix()


    def H(self, T):
        """
        Observation operator for model (linear interpolation)
        """
        return np.interp(self.data.z_obs[1:-1], self.z[1:-1], T, left=0, right=0)
    
    
    def compute_H_matrix(self):
        """
        Generates interpolation matrix between grid on which the observations
        are defined and the grid on which the problem is solved. 
        """
        H_matrix = np.zeros((self.data.M-1, self.N-1))
        T_test = np.zeros(self.N-1)
        for i in range(self.N-1):
            T_test[i] = 1
            H_matrix[:,i] = self.H(T_test)
            T_test[:] = 0
        return H_matrix
        
    
    
    def compute_dRdT(self, beta, T):
        main = 1+2*self.dz**2*self.model.eps_0*beta*T**3
        lower = -0.5*np.ones((self.N-2,))
        upper = -0.5*np.ones((self.N-2,))
        
        dRdT = np.diag(main)+np.diag(lower, k=-1)+np.diag(upper, k=1)
        return dRdT
    
    
    def compute_dRdbeta(self, T):        
        main = 0.5*self.dz**2*self.model.eps_0*(T**4-self.model.T_inf**4)
        dRdbeta = np.diag(main)
        return dRdbeta

    
    def compute_dJdT(self, beta):
        T, _ = self.model.get_T_mult(beta)
        
        dJdT = (self.H_matrix.dot(T)-self.data.T_obs).T.dot(np.linalg.inv(self.data.C_m)).dot(self.H_matrix)
        return dJdT
    
    
    def compute_dJdbeta(self, beta):
        dJdbeta = (beta-self.prior.mean).T.dot(self.prior.invcov)
        return dJdbeta
    
    
    # second derivatives
    def compute_dJdbetadbeta(self, beta):
        dJdbetadbeta = self.prior.invcov
        return dJdbetadbeta
    
    
    def compute_dRdTdbeta(self, T):
        dRdTdbeta = np.zeros((self.N-1, self.N-1, self.N-1))
        ind1, ind2, ind3 = np.diag_indices_from(dRdTdbeta)
        dRdTdbeta[ind1, ind2, ind3] = 2*self.dz**2*self.model.eps_0*T**3
        return dRdTdbeta
    
    
    def compute_dJdTdT(self):
        return self.H_matrix.T.dot(np.linalg.inv(self.data.C_m)).dot(self.H_matrix)
    
    
    def compute_dRdTdT(self, beta, T):
        dRdTdT = np.zeros((self.N-1, self.N-1, self.N-1))
        ind1, ind2, ind3 = np.diag_indices_from(dRdTdT)
        dRdTdT[ind1, ind2, ind3] = 6*T**2*self.dz**2*beta*self.model.eps_0
        return dRdTdT
    
    
    # cost function
    def compute_J(self, beta):
        T, _ = self.model.get_T_mult(beta)

        J = 0.5*(self.H_matrix.dot(T)-self.data.T_obs).T.dot(np.linalg.inv(self.data.C_m)).dot(self.H_matrix.dot(T)-self.data.T_obs)\
           +0.5*(beta-self.prior.mean).T.dot(self.prior.invcov).dot(beta-self.prior.mean)
        return J
    
    
    # adjoint equation
    def compute_psi(self, dRdT, dJdT):
        psi = np.linalg.solve(dRdT.T, -dJdT)
        return psi
    
    
    def compute_psi_cont(self, beta, T, dJdT):
        main = -2/self.dz-4*self.model.eps_0*self.dz*beta*T**3

        lower = (1/self.dz)*np.ones((self.N-2,))
        upper = (1/self.dz)*np.ones((self.N-2,))
        
        A = np.diag(main)+np.diag(lower, k=-1)+np.diag(upper, k=1)
        b = -dJdT 
        
        psi = np.linalg.solve(A, b)
        return psi


    # gradient
    def compute_gradient_adjoint(self, beta):
        T, _ = self.model.get_T_mult(beta)
        
        dRdT = self.compute_dRdT(beta, T)
        dRdbeta = self.compute_dRdbeta(T)
        dJdT = self.compute_dJdT(beta)
        dJdbeta = self.compute_dJdbeta(beta)

        psi = self.compute_psi(dRdT, dJdT)

        G = dJdbeta+psi.T.dot(dRdbeta)
        return G
    
    
    def compute_gradient_adjoint_cont(self, beta):
        T, _ = self.model.get_T_mult(beta)
        
        dJdT = self.compute_dJdT(beta)
        dJdbeta = self.compute_dJdbeta(beta)
        
        psi = self.compute_psi_cont(beta, T, dJdT)

        G = dJdbeta-self.model.eps_0*psi*self.dz*(T**4-self.model.T_inf**4)
        return G
        
        
    def compute_gradient_direct(self, beta):
        T, _ = self.model.get_T_mult(beta)
        
        dRdT = self.compute_dRdT(beta, T)
        dRdbeta = self.compute_dRdbeta(T)
        dJdT = self.compute_dJdT(beta)
        dJdbeta = self.compute_dJdbeta(beta)
        
        G = dJdbeta-dJdT.dot(np.linalg.inv(dRdT)).dot(dRdbeta)
        return G
        
        
    def compute_gradient_findiff(self, beta, epsilon=1e-5):
        G = np.zeros(self.N-1)
        Jbase = self.compute_J(beta)
        for i in range(self.N-1):
            beta[i] += epsilon             # perturb beta
            J = self.compute_J(beta)
            G[i] = (J - Jbase) / epsilon
            beta[i] -= epsilon             # restore beta
        return G


    # Hessian
    def compute_Hessian_findiff_check(self, beta):
        return Hessian(self.compute_J)(beta)
    
    
    def compute_Hessian_findiff(self, beta):
        H = np.zeros((self.N-1, self.N-1))
        dbeta = 1e-9
        for i in range(self.N-1):
            beta[i] -= dbeta
            dJdbeta_jm = self.compute_gradient_adjoint(beta)
            
            beta[i] += 2*dbeta
            dJdbeta_jp = self.compute_gradient_adjoint(beta)
            H[i,:] = (dJdbeta_jp - dJdbeta_jm)/(2*dbeta)
        beta[:] = beta[:]
        return H
        
    
    def compute_Hessian_adjoint_adjoint(self, beta):
        T, _ = self.model.get_T_mult(beta)
        
        dRdbeta = self.compute_dRdbeta(T)
        dRdT = self.compute_dRdT(beta, T)
        dJdT = self.compute_dJdT(beta)
        psi = self.compute_psi(dRdT, dJdT)
        
        dRdTdbeta = self.compute_dRdTdbeta(T)
        dJdTdT = self.compute_dJdTdT()
        dRdTdT = self.compute_dRdTdT(beta, T)
        dJdbetadbeta = self.compute_dJdbetadbeta(beta)
        
        nu = -dRdbeta.T.dot(np.linalg.inv(dRdT))
        
        mu = np.einsum('ik,mk->im', 
                       -np.einsum('m,mik->ik', psi, dRdTdbeta)-nu.dot(dJdTdT)\
                           -np.einsum('in,m,mnk->ik', nu, psi, dRdTdT),
                       np.linalg.inv(dRdT))
        
        H = dJdbetadbeta+mu.dot(dRdbeta)+np.einsum('in,m,mnj->ij', nu, psi, dRdTdbeta)
        return H


    def compute_Hessian_adjoint_direct(self, beta):
        T, _ = self.model.get_T_mult(beta)
        
        dRdbeta = self.compute_dRdbeta(T)
        dRdT = self.compute_dRdT(beta, T)
        dJdT = self.compute_dJdT(beta)
        psi = self.compute_psi(dRdT, dJdT)
        
        dRdTdbeta = self.compute_dRdTdbeta(T)
        dJdTdT = self.compute_dJdTdT()
        dRdTdT = self.compute_dRdTdT(beta, T)
        dJdbetadbeta = self.compute_dJdbetadbeta(beta)
        
        dTdbeta = -np.linalg.inv(dRdT).dot(dRdbeta)

        temp1 = np.einsum('nk,kj->nj', dJdTdT, dTdbeta)+\
                np.einsum('m,mnj->nj', psi, dRdTdbeta)+\
                np.einsum('m,mnk,kj->nj', psi, dRdTdT, dTdbeta)
        
        dpsidbeta = -np.einsum('nj,mn->mj', temp1, np.linalg.inv(dRdT))
        
        H = dJdbetadbeta+np.einsum('m,mik,kj->ij', psi, dRdTdbeta, dTdbeta)+\
                         np.einsum('mj,mi->ij', dpsidbeta, dRdbeta)
        assert np.allclose(dJdTdT, dJdTdT.T)
        return H


    def compute_Hessian_direct_adjoint(self, beta):
        T, _ = self.model.get_T_mult(beta)
        
        dRdbeta = self.compute_dRdbeta(T)
        dRdT = self.compute_dRdT(beta, T)
        dJdT = self.compute_dJdT(beta)
        psi = self.compute_psi(dRdT, dJdT)
        
        dRdTdbeta = self.compute_dRdTdbeta(T)
        dJdTdT = self.compute_dJdTdT()
        dRdTdT = self.compute_dRdTdT(beta, T)
        dJdbetadbeta = self.compute_dJdbetadbeta(beta)
        
        dTdbeta = np.linalg.solve(dRdT, -dRdbeta)
        
        H = dJdbetadbeta+np.einsum('km,ki,mj->ij', dJdTdT, dTdbeta, dTdbeta)\
                        +np.einsum('n,nkm,ki,mj->ij', psi, dRdTdT, dTdbeta, dTdbeta)\
                        +np.einsum('n,nik,kj->ij', psi, dRdTdbeta, dTdbeta)\
                        +np.einsum('n,nkj,ki->ij', psi, dRdTdbeta, dTdbeta)
        return H























