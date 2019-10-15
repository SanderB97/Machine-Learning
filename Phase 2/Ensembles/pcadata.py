# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:46:33 2018

@author: Olivier
"""
import helpers
import sklearn
from sklearn.decomposition import PCA


def get_pca_data(features, codebook, label_strings):
    train_data=[]
    train_labels=[]
    
    for image_features in features:
        bow_feature_vector = helpers.encodeImage(image_features.data,codebook)
        train_data.append(bow_feature_vector)
        train_labels.append(image_features.label)
              
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(label_strings)
    train_labels = label_encoder.transform(train_labels)
    
    pca = PCA(n_components = 150)
    PCA_features = pca.fit_transform(train_data)
    return PCA_features, train_labels, pca