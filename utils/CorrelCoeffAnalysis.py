import pandas as pd
import os
import numpy as np
import csv
from pandas.plotting import scatter_matrix
from datetime import datetime
from scipy.stats import pearsonr, betai
import matplotlib.pyplot as plt


class CorrelCoeffAnalysis(object):
    '''A class to help analyse the data;
    try to identify linear correlations in the data;
    calculate Pearson Correlation Coefficient with and without
    p-values; create a scatter matrix; inout data must not contain
    any strings or NaN values; also remove any columns with 
    categorical data or transform them first; remove any text except column labels'''
    def __init__(self, X_train):
        if X_train.isnull().any().any() == True:
            X_train = X_train.dropna(axis=1)
        pass
    
    
    def calculate_pearson_cc(X_train):
        '''This functions calculates a simple statistics of
        Pearson correlation coefficient'''
        attr = list(X_train)
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        for a in attr:
            corr_X_train = X_train.corr()        
            with open(os.path.join(METRIX_PATH, 'linear_PearsonCC_values_'+datestring+'.txt'), 'a') as text_file:
                corr_X_train[a].sort_values(ascending=False).to_csv(text_file)
        text_file.close()
    #calculate_pearson_cc(metrix)
    
    def print_scatter_matrix(X_train):
        '''A function to create a scatter matrix of the data'''
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        columns = [X_train.columns]
        for c in columns:
            if X_train.isnull().any().any() == True:
                X_train = X_train.dropna(axis=1)
        attr = list(X_train)    
        scatter_matrix(X_train[attr], figsize=(25,20))
        plt.savefig(os.path.join(METRIX_PATH, 'linear_PearsonCC_scattermatrix'+datestring+'.png'))
    #print_scatter_matrix(metrix)
               
    def corrcoef_loop(X_train):
        attr = list(X_train)
        rows = len(attr)
        r = np.ones(shape=(rows, rows))
        p = np.ones(shape=(rows, rows))
        for i in range(rows):
            for j in range(i+1, rows):
                c1 = X_train[attr[i]]
                c2 = X_train[attr[j]]
                r_, p_ = pearsonr(c1, c2)
                r[i, j] = r[j, i] = r_
                p[i, j] = p[j, i] = p_
        return r, p
    #r,p = corrcoef_loop(metrix_num)



    def write_corr(r, p, X_train):
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        with open(os.path.join(METRIX_PATH,
                           'linear_corr_pvalues_X_database_train_work'
                           +datestring+'.csv'), 'a') as csv_file:
            attr = list(X_train)
            rows = len(attr)
            for i in range(rows):
                for j in range(i+1, rows):
                    csv_file.write('%s, %s, %f, %f\n' 
                               %(attr[i], attr[j], r[i,j], p[i,j]))
            csv_file.close()
    #write_corr(r,p, metrix_num)



        
        
    #calculate_pearson_cc(metrix_num)
    #print_scatter_matrix(metrix_num)
    #r,p = corrcoef_loop(metrix_num)
    #write_corr(r,p, metrix_num)
#CorrelCoeffAnalysis(metrix)
        
        