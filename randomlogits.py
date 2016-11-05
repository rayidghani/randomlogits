from __future__ import division
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
import pylab as pl
from itertools import chain

def TrainRandomLogits(X, y, n_logits, n_features):
    clf = BaggingClassifier(base_estimator = LogisticRegression(),
        n_estimators=n_logits, max_features = n_features)
    clf.fit(X, y)
    return clf


def GetFeatureImportances(rl):
    num_logits = rl.get_params(deep=False)['n_estimators']
    array = rl.estimators_features_
    total_features = len(set(chain(*array)))
    n_features = rl.get_params(deep=False)['max_features']
	
    #initialize feature importance matrix
    feature_importance_matrix = np.zeros((num_logits, total_features))
    row = 0
    for feature in rl.estimators_features_:
    	for i in range(n_features):
    		feature_importance_matrix[row][feature[i]] = rl.estimators_[row].coef_[0][i] 
    	row += 1
    feature_importance_matrix[feature_importance_matrix==0]=['nan']
    mean_feature_importance = np.nanmean(feature_importance_matrix, dtype=np.float64, axis=0)
    std_feature_importance = np.nanstd(feature_importance_matrix, dtype=np.float64,axis=0)

    return (feature_importance_matrix,mean_feature_importance, std_feature_importance)


