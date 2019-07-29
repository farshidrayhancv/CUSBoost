import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy import interp
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.model_selection import StratifiedKFold




def cus_sampler(X_train, y_train, number_of_clusters=23, percentage_to_choose_from_each_cluster=0.50):
    """
    number_of_clusters = 23
    percentage_to_choose_from_each_cluster: 50%
    """

    selected_idx = []
    selected_idx = np.asarray(selected_idx)

    value, counts = np.unique(y_train, return_counts=True)
    minority_class = value[np.argmin(counts)]
    majority_class = value[np.argmax(counts)]

    idx_min = np.where(y_train == minority_class)[0]
    idx_maj = np.where(y_train == majority_class)[0]

    majority_class_instances = X_train[idx_maj]
    majority_class_labels = y_train[idx_maj]

    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(majority_class_instances)

    X_maj = []
    y_maj = []

    points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    for key in points_under_each_cluster.keys():

        points_under_this_cluster = np.array(points_under_each_cluster[key])
        number_of_points_to_choose_from_this_cluster = math.ceil(
            len(points_under_this_cluster) * percentage_to_choose_from_each_cluster)




        selected_points = np.random.choice(points_under_this_cluster,
                                           size=number_of_points_to_choose_from_this_cluster, replace=False)
        X_maj.extend(majority_class_instances[selected_points])
        y_maj.extend(majority_class_labels[selected_points])

        selected_idx = np.append(selected_idx,selected_points)

        # print(len(selected_idx))

        selected_idx = selected_idx.astype(int)


    X_sampled = X_train[selected_idx]
    y_sampled = y_train[selected_idx]


    # X_sampled = np.concatenate((X_train[idx_min], np.array(X_maj)))
    # y_sampled = np.concatenate((y_train[idx_min], np.array(y_maj)))

    # print(X_sampled.shape, y_sampled.shape, selected_idx.shape)

    return X_sampled, y_sampled, selected_idx