# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:35:02 2018

@author: Olivier
"""

import helpers
import pickle
import os

OUTPUT_PATH = './'
FEATURE_TRAIN_PATH = os.path.join(OUTPUT_PATH,'features_train')
FILEPATTERN_DESCRIPTOR_TRAIN = os.path.join(FEATURE_TRAIN_PATH,'train_features_{}.pkl')

descriptor_desired='sift'
with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired),'rb') as pkl_file_train:
    sift_features=pickle.load(pkl_file_train)
    
descriptor_desired='boost_desc'
with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired),'rb') as pkl_file_train:
    desc_features=pickle.load(pkl_file_train)
    
descriptor_desired='daisy'
with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired),'rb') as pkl_file_train:
    daisy_features=pickle.load(pkl_file_train)
    
descriptor_desired='freak'
with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired),'rb') as pkl_file_train:
    freak_features=pickle.load(pkl_file_train)
    
descriptor_desired='lucid'
with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired),'rb') as pkl_file_train:
    lucid_features=pickle.load(pkl_file_train)
    
descriptor_desired='orb'
with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired),'rb') as pkl_file_train:
    orb_features=pickle.load(pkl_file_train)
    
descriptor_desired='vgg'
with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired),'rb') as pkl_file_train:
    vgg_features=pickle.load(pkl_file_train)
    

def get_codebooks():
    sift_codebook = helpers.createCodebook(sift_features, codebook_size = 500)
#    desc_codebook = helpers.createCodebook(desc_features, codebook_size = 500)
#    daisy_codebook = helpers.createCodebook(daisy_features, codebook_size = 500)
#    freak_codebook = helpers.createCodebook(freak_features, codebook_size = 500)
#    lucid_codebook = helpers.createCodebook(lucid_features, codebook_size = 500)
#    orb_codebook = helpers.createCodebook(orb_features, codebook_size = 500)
#    vgg_codebook = helpers.createCodebook(vgg_features, codebook_size = 500)
    return sift_codebook #, desc_codebook, daisy_codebook, freak_codebook, lucid_codebook, orb_codebook, vgg_codebook