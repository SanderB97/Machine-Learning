"""Plot learning curve"""
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    titlepar = {'weight' : 'bold',
            'size' : '15'}
    ypara = {'weight' : 'bold',
             'rotation' : '0'}
    xpara = {'weight' : 'bold'}
    plt.title(title, **titlepar)
    plt.xlabel("Amount of training images", **xpara)
    plt.ylabel("Score", **ypara).set_position([-1.5, 0.45])
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross validation score")
    plt.legend(loc="best")
    return plt

"""Plot ROC"""
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def plotROC(x, y, nameofTest, test, cv, label_strings):
    y_bin = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9,10])
    n_classes = y_bin.shape[1]
    
    pipeline= Pipeline([('scaler', StandardScaler()), (nameofTest, test)])
    y_score = cross_val_predict(pipeline, x, y, cv=cv ,method='predict_proba')
    f = dict()
    t = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        f[i], t[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(f[i], t[i])
    colors =(['blue', 'red', 'green','yellow', 'black','coral','orange','cyan','peachpuff','deeppink','forestgreen'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(f[i], t[i], color=color, lw=3,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                 ''.format(label_strings[i], roc_auc[i]))
        
    plt.plot([0, 1], [0, 1], 'k--', lw=3)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    titlepar = {'weight' : 'bold',
            'size' : '15'}
    ypara = {'weight' : 'bold',
             'rotation' : '0',}
    xpara = {'weight' : 'bold'}
    plt.xlabel('False Positive Rate', **xpara)
    plt.ylabel('True Positive \n Rate', **ypara).set_position([-1, 0.43])
    plt.title('ROC for '+nameofTest, **titlepar)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()

"""Confusion matrix"""
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_conf_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    titlepar = {'weight' : 'bold',
            'size' : '20'}
    ypara = {'weight' : 'bold',
             'rotation' : '0',
             'size' : '14'}
    xpara = {'weight' : 'bold',
             'size' : '14'}
    plt.ylabel('True label', **ypara)
    plt.xlabel('Predicted label', **xpara)
    plt.title(title, **titlepar)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, weight='bold')
    plt.yticks(tick_marks, classes, weight = 'bold')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    return plt.tight_layout()

def plot_confusion_matrix(model, x_train, y_train, x_test, y_test, label_strings):
    model.fit(x_train, y_train)
    y_true = y_test
    y_pred = model.predict(x_test)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure(figsize = (20, 20))
    plot_conf_matrix(cnf_matrix, classes=label_strings,
                          title='Confusion matrix (%)')
    
    # Plot normalized confusion matrix
    plt.figure(figsize = (20, 20))
    return plot_conf_matrix(cnf_matrix, classes=label_strings, normalize=True,
                          title='Normalized confusion matrix')


"""Plot boxplot of average cv_score of some models"""

"""different models to test are saved in models[], see example below"""
#modelsList = []
#modelsList.append(('Logistic Regression', linear_model.LogisticRegression()))
#modelsList.append(('Logistic Regression CV implemented', linear_model.LogisticRegressionCV()))
#modelsList.append(('OneVsRestClassifier on Logistic Regression', OneVsRestClassifier(linear_model.LogisticRegression())))
#modelsList.append(('Linear SVC', svm.LinearSVC()))
#modelsList.append(('LDA', linear_model.LinearDiscriminantAnalysis()))

#def first_impression_model_comparison(modelsList, x_train, y_train, cv):
#    results=[]
#    names=[]
#    scoring = 'accuracy'
#    for name, model in modelsList:
#        cv_results = cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring)
#        results.append(cv_results)
#        names.append(name)
#    # boxplot algorithm comparison
#    fig = plt.figure()
#    fig.set_size_inches(10, 10)
#    ax = fig.add_subplot(111)
#    plt.boxplot(results)
#    ax.set_xticklabels(names)
#    titlepar = {'weight' : 'bold',
#            'size' : '15'}
#    ypara = {'weight' : 'bold',
#             'rotation' : '0',}
#    plt.ylabel("Cross validation score", **ypara).set_position([-1.5, 1.01])
#    plt.title("Cross validation score per model", **titlepar)
#    plt.ylim(ymin = 0.30, ymax = 0.50)
#    xpara = {'rotation' : '40',
#             'ha' : 'right'}