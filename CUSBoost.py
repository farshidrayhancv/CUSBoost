# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:54:20 2017

@author: Farshid Rayhan, United International University

"""

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from cus_sampling import cus_sampler
from sklearn.cluster import KMeans
from sklearn.ensemble.weight_boosting import _samme_proba
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


class CUSBoostClassifier:
    def __init__(self, n_estimators, depth):
        self.M = n_estimators
        self.depth = depth
        self.undersampler = RandomUnderSampler(return_indices=True,replacement=False)

        ## Some other samplers to play with ######
        # self.undersampler = EditedNearestNeighbours(return_indices=True,n_neighbors=neighbours)
        # self.undersampler = AllKNN(return_indices=True,n_neighbors=neighbours,n_jobs=4)

    def fit(self, X, Y):
        self.models = []
        self.alphas = []

        N, _ = X.shape
        W = np.ones(N) / N

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')


            X_undersampled, y_undersampled, chosen_indices = cus_sampler(X,Y)


            tree.fit(X_undersampled, y_undersampled,
                     sample_weight=W[chosen_indices])

            P = tree.predict(X)

            err = np.sum(W[P != Y])

            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:
                try:
                    if (np.log(1 - err) - np.log(err)) == 0 :
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))
                    W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1
                except:
                    alpha = 0
                    # W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1

                self.models.append(tree)
                self.alphas.append(alpha)

    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha * tree.predict(X)
        return np.sign(FX), FX

    def predict_proba(self, X):
        # if self.alphas == 'SAMME'
        proba = sum(tree.predict_proba(X) * alpha for tree , alpha in zip(self.models,self.alphas) )


        proba = np.array(proba)


        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        # proba =  np.linspace(proba)
        # proba = np.array(proba).astype(float)
        proba = proba /  normalizer

        # print(proba)
        return proba

    def predict_proba_samme(self, X):
        # if self.alphas == 'SAMME.R'
        proba = sum(_samme_proba(est , 2 ,X) for est in self.models )

        proba = np.array(proba)

        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        # proba =  np.linspace(proba)
        # proba = np.array(proba).astype(float)
        proba = proba / normalizer

        # print('proba = ',proba)
        return proba.astype(float)
