#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:31:14 2018

@author: arent
"""

import numpy as np


class Prior:
    """
    Prior distribution of the corrective term. 

    Args:
        mean: mean of the prior distribution
        cov: prior covariance matrix

    Returns:
        Prior definition.
    """
    
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.invcov = np.linalg.inv(cov)









