import numpy as np
from scipy.io import loadmat
import collections
import math
from collections import OrderedDict
import pdb
# Data Requirements:
# Remove self loops
# Arrange user pairs in chronological order
# Remove x% of old users "O" to predict
# Scale to smaller dataset size for testing


class data_handler():

    def __init__(self,rating_path,trust_path,time_path):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.time_path = time_path
        f1 = open(rating_path)
        P = loadmat(f1)
        P = P['rating_with_timestamp']
        self.n = max(P[:,0])
        self.k = 0
        self.d = 0

    

    def load_matrices(self):
        #Loading matrices from data
        f2 = open(self.trust_path)
        G_raw = loadmat(f2) #trust-trust matrices
        G_raw = G_raw['trust']
        print G_raw.shape

        print "CALCING G"

        #constructing initial trust matrix G and setting all users in newer_users to 0
        G_needed = np.zeros((self.n,self.n), dtype = np.float32)
        
        #db.set_trace()
        G_raw = G_raw - 1
        G_needed[G_raw[:, 0], G_raw[:, 1]] = 1

        return [None, None, None, G_needed]

data = data_handler("rating_with_timestamp.mat", "trust.mat", "epinion_trust_with_timestamp.mat")
data.load_matrices()








