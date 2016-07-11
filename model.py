#! /usr/bin/env python

from __future__ import print_function
import numpy as np
from theano import tensor as T
import theano
from scipy.sparse import coo_matrix
import pdb
from scipy.io import loadmat
from data_handler_hTrust import data_handler
theano.config.compute_test_value = 'off'

class AutoEncoder(object):

    def __init__(self, path, k):
        # hidden layer's dimension
        #pdb.set_trace()
        data = data_handler("rating_with_timestamp.mat", "trust.mat", "epinion_trust_with_timestamp.mat")
        _, _, _, self.T = data.load_matrices()

        self.n = self.T.shape[1]
        self.k = k
        self.W = []
        self.V = []
        self.b = []
        self.mu = []

    def model(self, loss='bce', lr=0.001):
        ''' define the AutoEncoder model (mini-batch) '''
        # initializing network paramters
        w = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.n, self.k)).astype(np.float32)
        v = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.k, self.n)).astype(np.float32)
        MU = np.zeros((self.k,1)).astype(np.float32)
        B = np.zeros((self.n,1)).astype(np.float32)
        # Creating theano shared variables from these
        W = theano.shared(w, name='W')#, borrow=True)
        V = theano.shared(v, name='V') #,borrow=True)
        mu = theano.shared(MU, name='mu')#, borrow=True)
        b = theano.shared(B, name='b') #, borrow=True)
        self.W = W
        self.V = V
        self.b = b
        self.mu = mu
        self.param = [W, V, mu, b]
        rating = T.matrix() # Assuming d * b input
        temp = T.dot(self.V, rating.T)
        G = T.nnet.sigmoid(T.dot(self.V, rating.T) + self.mu)
        self.debug = theano.function([rating], temp)
        F = T.nnet.sigmoid(T.dot(self.W, G) + self.b)
        
        self.loss = T.sum((F - rating) ** 2) + \
            0.001 * T.sum(W ** 2) + 0.001 * T.sum(V ** 2)
        grads = T.grad(self.loss, self.param)
        updates = [(param, param - lr * grad) for (param, grad) in \
                   zip(self.param, grads)]

        self.ae = theano.function([rating], self.loss, updates=updates)
        #self.debug = theano.function([rating], scan_res, updates=None)

    def get_params(self):
        return self.W.get_value(), self.V.get_value(), self.b.get_value(), self.mu.get_value()

if __name__ == "__main__":
    AE = AutoEncoder('../data/data.mat', 100)
    AE.model_batch()
    rating = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)
    AE.ae_batch(rating)
    #x = AE.debug(rating)
    pdb.set_trace()
