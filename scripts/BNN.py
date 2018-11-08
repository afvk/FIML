#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:48:10 2018

@author: arent
"""



import numpy as np

x_train = np.linspace(-3, 3, num=50)
y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=50)
x_train = x_train.astype(np.float32).reshape((50, 1))
y_train = y_train.astype(np.float32).reshape((50, 1))

import tensorflow as tf
import tensorflow_probability as tfp
#from edward.models import Normal

from tensorflow_probability import edward2 as ed


W_0 = ed.Normal(loc=tf.zeros([1, 2]), scale=tf.ones([1, 2]))
W_1 = ed.Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]))
b_0 = ed.Normal(loc=tf.zeros(2), scale=tf.ones(2))
b_1 = ed.Normal(loc=tf.zeros(1), scale=tf.ones(1))

x = x_train
y = ed.Normal(loc=tf.matmul(tf.tanh(tf.matmul(x, W_0) + b_0), W_1) + b_1,
           scale=0.1)

#
#qW_0 = ed.Normal(loc=tf.get_variable("qW_0/loc", [1, 2]),
#              scale=tf.nn.softplus(tf.get_variable("qW_0/scale", [1, 2])))
#qW_1 = ed.Normal(loc=tf.get_variable("qW_1/loc", [2, 1]),
#              scale=tf.nn.softplus(tf.get_variable("qW_1/scale", [2, 1])))
#qb_0 = ed.Normal(loc=tf.get_variable("qb_0/loc", [2]),
#              scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [2])))
#qb_1 = ed.Normal(loc=tf.get_variable("qb_1/loc", [1]),
#              scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [1])))


#inference = ed.KLqp({W_0: qW_0, W_1: qW_1, b_0: qb_0, b_1: qb_1},
#                    data={y: y_train})

def deep_exponential_family_variational():
    qW_0 = ed.Normal(loc=tf.get_variable("qW_0/loc", [1, 2]),
                  scale=tf.nn.softplus(tf.get_variable("qW_0/scale", [1, 2])))
    qW_1 = ed.Normal(loc=tf.get_variable("qW_1/loc", [2, 1]),
                  scale=tf.nn.softplus(tf.get_variable("qW_1/scale", [2, 1])))
    qb_0 = ed.Normal(loc=tf.get_variable("qb_0/loc", [2]),
                  scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [2])))
    qb_1 = ed.Normal(loc=tf.get_variable("qb_1/loc", [1]),
                  scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [1])))
    return qW_0, qW_1, qb_0, qb_1


def make_value_setter(**model_kwargs):
    """Creates a value-setting interceptor."""
    def set_values(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]
        return ed.interceptable(f)(*args, **kwargs)
    return set_values

# Compute expected log-likelihood. First, sample from the variational
# distribution; second, compute the log-likelihood given the sample.
qw2, qw1, qw0, qz2, qz1, qz0 = deep_exponential_family_variational()

with ed.tape() as model_tape:
    with ed.interception(make_value_setter(w2=qw2, w1=qw1, w0=qw0,
                                         z2=qz2, z1=qz1, z0=qz0)):
        posterior_predictive = deep_exponential_family(data_size, feature_size, units, shape)

log_likelihood = posterior_predictive.distribution.log_prob(bag_of_words)

# Compute analytic KL-divergence between variational and prior distributions.
kl = 0.
for rv_name, variational_rv in [("z0", qz0), ("z1", qz1), ("z2", qz2),
                            ("w0", qw0), ("w1", qw1), ("w2", qw2)]:
    kl += tf.reduce_sum(variational_rv.distribution.kl_divergence(
            model_tape[rv_name].distribution))

elbo = tf.reduce_mean(log_likelihood - kl)
tf.summary.scalar("elbo", elbo)
optimizer = tf.train.AdamOptimizer(1e-3)
train_op = optimizer.minimize(-elbo)




