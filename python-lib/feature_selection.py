# -*- coding: utf-8 -*-
from sklearn.feature_selection import SelectPercentile, f_classif,f_regression,chi2
from dataiku.customrecipe import *

def univariate_feature_selection(mode,predictors,target):
    
    if mode == 'f_regression':
        fselect = SelectPercentile(f_regression, 100)
        
    if mode == 'f_classif':
        fselect = SelectPercentile(f_classif, 100)
        
    if mode == 'chi2':
        fselect = SelectPercentile(chi2, 100)
        
    fselect.fit_transform(predictors, target)
    
    return fselect.pvalues_