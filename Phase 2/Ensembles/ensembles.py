# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:34:47 2018

@author: Olivier
"""
import codebooks as cb
import pcadata
import numpy as np
import os
import glob
import sklearn.ensemble as es
import helpers
import pcatestdata as ptd
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn import linear_model
import Graphs as gr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.tree as st
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


DATA_BASE_PATH = './'
OUTPUT_PATH = './'
DATA_TRAIN_PATH = os.path.join(DATA_BASE_PATH,'train')
PREDICTION_PATH = os.path.join(OUTPUT_PATH,'predictions')
DATA_BASE_PATH = './'
DATA_TEST_PATH = os.path.join(DATA_BASE_PATH,'test')

folder_paths=glob.glob(os.path.join(DATA_TRAIN_PATH,'*'))
label_strings = np.sort(np.array([os.path.basename(path) for path in folder_paths]))
num_classes = label_strings.shape[0]

#sift_codebook, desc_codebook, daisy_codebook, freak_codebook, lucid_codebook, orb_codebook, vgg_codebook = cb.get_codebooks()
sift_codebook = cb.get_codebooks()

sift_pca, sift_labels, pca_sift = pcadata.get_pca_data(cb.sift_features, sift_codebook, label_strings)
#desc_pca, desc_labels, pca_desc = pcadata.get_pca_data(cb.desc_features, desc_codebook, label_strings)
#daisy_pca, daisy_labels, pca_daisy = pcadata.get_pca_data(cb.daisy_features, daisy_codebook, label_strings)
#freak_pca, freak_labels, pca_freak = pcadata.get_pca_data(cb.freak_features, freak_codebook, label_strings)
#lucid_pca, lucid_labels, pca_lucid = pcadata.get_pca_data(cb.lucid_features, lucid_codebook, label_strings)
#orb_pca, orb_labels, pca_orb = pcadata.get_pca_data(cb.orb_features, orb_codebook, label_strings)
#vgg_pca, vgg_labels, pca_vgg = pcadata.get_pca_data(cb.vgg_features, vgg_codebook, label_strings)

#all_features = np.concatenate((np.concatenate((np.concatenate((np.concatenate((np.concatenate((np.concatenate((sift_pca,desc_pca),axis=1),daisy_pca),axis=1),freak_pca),axis=1),lucid_pca),axis=1),orb_pca),axis=1),vgg_pca),axis=1)
all_features = sift_pca
all_labels = sift_labels #does not matter

X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.25, random_state=42)

bag = es.BaggingClassifier(base_estimator=st.DecisionTreeClassifier(max_depth=9),n_estimators=75)
ada = es.AdaBoostClassifier(base_estimator=st.DecisionTreeClassifier(max_depth=6),n_estimators=185)
et = es.ExtraTreesClassifier(n_estimators=90, max_depth=18)
rf = es.RandomForestClassifier(n_estimators=250,max_depth=15)
knn = KNeighborsClassifier(n_neighbors=190, weights='uniform')

voter5 = es.VotingClassifier(estimators=[('bag',bag),('ada',ada),('et',et),('rf',rf),('knn',knn)],voting='soft')
voter2 = es.VotingClassifier(estimators=[('rf',rf),('knn',knn)],voting='soft')
#voter5.fit(all_features, all_labels)
#voter2.fit(all_features, all_labels)
#knn.fit(all_features, all_labels)
#models_list = [bag, ada, et, rf, knn]
#train_meta = np.empty((all_features.shape[0],11*len(models_list)))
#
#skf = StratifiedKFold(n_splits=5)
#for train_index, test_index in skf.split(all_features, all_labels):
#    train_data = all_features[train_index]
#    train_labels = all_labels[train_index]
#    test_data = all_features[test_index]
#    for i in range(len(models_list)):
#        model = models_list[i]
#        model.fit(train_data, train_labels)
#        pred = model.predict_proba(test_data)
#        train_meta[test_index,i*11:(i+1)*11] = pred

#neighbors_range = [120,140,160,100,180,110,130,150,170,190,200,210,220,230,240,250,260,270,280,290,300]
#base_estimators = [st.DecisionTreeClassifier(max_depth=6)]
#n_estimators_range = [260,270,240,230,220]
#criteria = ["gini", "entropy"]
#weights = ["uniform", "distance"]
#depths = [15]
#learning_rates = [.5,1,1.5]
#
#params = {'n_neighbors': neighbors_range, 'weights': weights}
#grid = GridSearchCV(estimator=knn,param_grid=params,cv=5,scoring='neg_log_loss')
#grid.fit(all_features, all_labels)
#print(grid.best_score_)
#print(grid.best_params_)

#lc_bag = gr.plot_learning_curve(bag, "Bagging classifier", all_features, all_labels)
#lc_bag.savefig('lcbag')
#lc_ada = gr.plot_learning_curve(ada, "AdaBoost classifier", all_features, all_labels)
#lc_ada.savefig('lcada')
#lc_et = gr.plot_learning_curve(et, "Extra trees classifier", all_features, all_labels)
#lc_et.savefig('lcet')
#lc_rf = gr.plot_learning_curve(rf, "Random forest classifier", all_features, all_labels)
#lc_rf.savefig('lcrf')
#lc_knn = gr.plot_learning_curve(knn, "KNN classifier", all_features, all_labels)
#lc_knn.savefig('lcknn')

#knn.fit(X_train, y_train)
#bag.fit(X_train, y_train)
#ada.fit(X_train, y_train)
#et.fit(X_train, y_train)
#rf.fit(X_train, y_train)
#
#predknn = knn.predict_proba(X_train)
#predbag = bag.predict_proba(X_train)
#predada = ada.predict_proba(X_train)
#predet = et.predict_proba(X_train)
##predrf = rf.predict_proba(X_train)
#predvot5 = voter5.predict_proba(X_train)
#predvot2 = voter2.predict_proba(X_train)
#predknntest = knn.predict_proba(X_test)
#predbagtest = bag.predict_proba(X_test)
#predadatest = ada.predict_proba(X_test)
#predettest = et.predict_proba(X_test)
#predrftest = rf.predict_proba(X_test)
#print(sklearn.metrics.log_loss(y_train, predknn))
#print(sklearn.metrics.log_loss(y_train, predbag))
#print(sklearn.metrics.log_loss(y_train, predada))
#print(sklearn.metrics.log_loss(y_train, predet))
#print(sklearn.metrics.log_loss(y_train, predrf))
#print(sklearn.metrics.log_loss(y_train, predvot5))
#print(sklearn.metrics.log_loss(y_train, predvot2))
#print(sklearn.metrics.log_loss(y_test, predknntest))
#print(sklearn.metrics.log_loss(y_test, predbagtest))
#print(sklearn.metrics.log_loss(y_test, predadatest))
#print(sklearn.metrics.log_loss(y_test, predettest))
#print(sklearn.metrics.log_loss(y_test, predrftest))

#conf_bag = gr.plot_confusion_matrix(bag, X_train, y_train, X_test, y_test, label_strings)
#conf_ada = gr.plot_confusion_matrix(ada, X_train, y_train, X_test, y_test, label_strings)
#conf_et = gr.plot_confusion_matrix(et, X_train, y_train, X_test, y_test, label_strings)
#conf_rf = gr.plot_confusion_matrix(rf, X_train, y_train, X_test, y_test, label_strings)
#conf_knn = gr.plot_confusion_matrix(knn, X_train, y_train, X_test, y_test, label_strings)
conf_vot2 = gr.plot_confusion_matrix(voter2, X_train, y_train, X_test, y_test, label_strings)
conf_vot5 = gr.plot_confusion_matrix(voter5, X_train, y_train, X_test, y_test, label_strings)

#roc_bag = gr.plotROC(all_features, all_labels, "Bagging classifier", bag, 5, label_strings)
#roc_ada = gr.plotROC(all_features, all_labels, "AdaBoost classifier", ada, 5, label_strings)
#roc_et = gr.plotROC(all_features, all_labels, "Extra trees classifier", et, 5, label_strings)
#roc_rf = gr.plotROC(all_features, all_labels, "Random forest classifier", rf, 5, label_strings)
#roc_knn = gr.plotROC(all_features, all_labels, "KNN classifier", knn, 5, label_strings)

#
#test_sift = ptd.get_pca(ptd.sift_features,sift_codebook,pca_sift)
#test_desc = ptd.get_pca(ptd.desc_features,desc_codebook,pca_desc)
#test_daisy = ptd.get_pca(ptd.daisy_features,daisy_codebook,pca_daisy)
#test_freak = ptd.get_pca(ptd.freak_features,freak_codebook,pca_freak)
#test_lucid = ptd.get_pca(ptd.lucid_features,lucid_codebook,pca_lucid)
#test_orb = ptd.get_pca(ptd.orb_features,orb_codebook,pca_orb)
#test_vgg = ptd.get_pca(ptd.vgg_features,vgg_codebook,pca_vgg)
#
#test_features = np.concatenate((np.concatenate((np.concatenate((np.concatenate((np.concatenate((np.concatenate((test_sift,test_desc),axis=1),test_daisy),axis=1),test_freak),axis=1),test_lucid),axis=1),test_orb),axis=1),test_vgg),axis=1)

#test_meta = np.empty((test_sift.shape[0],11*len(models_list)))
#
#for i in range(len(models_list)):
#    model = models_list[i]
#    model.fit(all_features, all_labels)
#    pred = model.predict_proba(test_sift)
#    test_meta[:,i*11:(i+1)*11] = pred
#    
#meta_model = linear_model.LogisticRegressionCV(Cs=5,cv=5)
#meta_model.fit(train_meta, all_labels)
#X = meta_model.predict_proba(test_meta)
#X2 = meta_model.predict_proba(train_meta)
#print(metrics.log_loss(y_test, X))
#print(metrics.log_loss(y_train, X2))
#X = voter.predict_proba(test_sift)
#true_labels = []
#print(metrics.log_loss(true_labels, X))
#
# Build a submission
#pred_file_path = os.path.join(PREDICTION_PATH, helpers.generateUniqueFilename('voting5','csv'))
#helpers.writePredictionsToCsv(X,pred_file_path,label_strings)