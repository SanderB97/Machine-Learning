# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:32:39 2018

@author: Olivier
"""
import os
import pickle
import helpers
from sklearn.decomposition import PCA

OUTPUT_PATH = './'
FEATURE_TEST_PATH = os.path.join(OUTPUT_PATH,'features_test')
FILEPATTERN_DESCRIPTOR_TEST = os.path.join(FEATURE_TEST_PATH,'test_features_{}.pkl')

descriptor_desired='sift'
with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desired),'rb') as pkl_file_test:
    sift_features=pickle.load(pkl_file_test)
    
descriptor_desired='boost_desc'
with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desired),'rb') as pkl_file_test:
    desc_features=pickle.load(pkl_file_test)
    
descriptor_desired='daisy'
with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desired),'rb') as pkl_file_test:
    daisy_features=pickle.load(pkl_file_test)
    
descriptor_desired='freak'
with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desired),'rb') as pkl_file_test:
    freak_features=pickle.load(pkl_file_test)
    
descriptor_desired='lucid'
with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desired),'rb') as pkl_file_test:
    lucid_features=pickle.load(pkl_file_test)
    
descriptor_desired='orb'
with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desired),'rb') as pkl_file_test:
    orb_features=pickle.load(pkl_file_test)
    
descriptor_desired='vgg'
with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desired),'rb') as pkl_file_test:
    vgg_features=pickle.load(pkl_file_test)
      
def get_pca(features, codebook, pca):         
    test_data=[]
    for image_features in features:
        bow_feature_vector=helpers.encodeImage(image_features.data,codebook)
        test_data.append(bow_feature_vector)
    return pca.transform(test_data)

