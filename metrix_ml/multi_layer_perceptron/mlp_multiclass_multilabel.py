import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import os
import seaborn as sns
import sklearn
import joblib
import keras
import tensorflow

from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std

print("Pandas version: ", pd.__version__)
print("Numpy version: ", np.__version__)
print("Matplotlib.pyplot version: ", matplotlib.__version__)
print("Seaborn version: ", sns.__version__)
print("SciKitLearn version: ", sklearn.__version__)
print("Keras version: ", keras.__version__)
print("TensorFlow version: ", tensorflow.__version__)


# Find the input data as CSV
run1_3dii_csv = "/dls/metrix/metrix/data_analysis/for_METRIX/test_metrix_ml/20210426/data/summary_table_3dii_allData_allXtals_run1_BigEPruns.csv"
run2_3dii_csv = "/dls/metrix/metrix/data_analysis/for_METRIX/test_metrix_ml/20210426/data/summary_table_3dii_allData_allXtals_run2_BigEPruns.csv"
run3_3dii_csv = "/dls/metrix/metrix/data_analysis/for_METRIX/test_metrix_ml/20210426/data/summary_table_3dii_allData_allXtals_run3_BigEPruns.csv"
run4_3dii_csv = "/dls/metrix/metrix/data_analysis/for_METRIX/test_metrix_ml/20210426/data/summary_table_3dii_allData_allXtals_run4_BigEPruns.csv"


run1_dials_csv = "/dls/metrix/metrix/data_analysis/for_METRIX/test_metrix_ml/20210426/data/summary_table_dials_allData_allXtals_run1_BigEPruns.csv"
run2_dials_csv = "/dls/metrix/metrix/data_analysis/for_METRIX/test_metrix_ml/20210426/data/summary_table_dials_allData_allXtals_run2_BigEPruns.csv"
run3_dials_csv = "/dls/metrix/metrix/data_analysis/for_METRIX/test_metrix_ml/20210426/data/summary_table_dials_allData_allXtals_run3_BigEPruns.csv"
run4_dials_csv = "/dls/metrix/metrix/data_analysis/for_METRIX/test_metrix_ml/20210426/data/summary_table_dials_allData_allXtals_run4_BigEPruns.csv"


# Load the CSV data into a Pandas dataframe
run1_3dii_df = pd.read_csv(run1_3dii_csv)
run2_3dii_df = pd.read_csv(run2_3dii_csv)
run3_3dii_df = pd.read_csv(run3_3dii_csv)
run4_3dii_df = pd.read_csv(run4_3dii_csv)


run1_dials_df = pd.read_csv(run1_dials_csv)
run2_dials_df = pd.read_csv(run2_dials_csv)
run3_dials_df = pd.read_csv(run3_dials_csv)
run4_dials_df = pd.read_csv(run4_dials_csv)


df_3dii = pd.concat([run1_3dii_df, run2_3dii_df, run3_3dii_df, run4_3dii_df],
               ignore_index=True)
df_run234_3dii = pd.concat([run2_3dii_df, run3_3dii_df, run4_3dii_df],
               ignore_index=True)
#print(df_3dii.describe)

dials_df = pd.concat([run1_dials_df, run2_dials_df, run3_dials_df, run4_dials_df],
               ignore_index=True)
dials_run234_df = pd.concat([run2_dials_df, run3_dials_df, run4_dials_df],
               ignore_index=True)
#print(dials_df.describe)


full_3dii_df = df_3dii[['anomalousCC', 'lowreslimit', 'anomalousslope', 'diffF',
       'diffI', 'f', 'autobuild_success', 'crank2_success', 'autosharp_success']]

full_dials_df = dials_df[['anomalousCC', 'lowreslimit', 'anomalousslope', 'diffF',
       'diffI', 'f', 'autobuild_success', 'crank2_success', 'autosharp_success']]


# train-test-split
# this should be done stratified for multi-class multi-lable; the only
# support I found is in http://scikit.ml which is no longer maintained
# (as of 2018); I stick to normal train-test-split for now, also considering
# that in on ecase I have only one sample available; use the standard
# sklearn function and apply stratification

# train-test-split for 3dii
y_3dii = full_3dii_df[['autobuild_success', 'crank2_success', 'autosharp_success']]
X_3dii = full_3dii_df.iloc[:, :-4]

X_3dii_train, X_3dii_test, y_3dii_train, y_3dii_test = train_test_split(X_3dii, y_3dii,
                                                                        test_size=0.2,
                                                                        random_state=42,
                                                                        stratify=y_3dii)
print("Training X shape: ", X_3dii_train.shape)
print("Testing X shape: ", X_3dii_test.shape)
print("Training y shape: ", y_3dii_train.shape)
print("Training y shape: ", y_3dii_test.shape)
print(X_3dii_train[:10], y_3dii_train[:10])

# train-test-split for DIALS
# needed to drop the sample for which there is only one occurance in order
# to have train-test-split run
full_dials_df = full_dials_df.drop(full_dials_df[(full_dials_df['autobuild_success']==0) &
                                                 (full_dials_df['crank2_success']==0) &
                                                 (full_dials_df['autosharp_success']>0)].index)

y_dials = full_dials_df[['autobuild_success', 'crank2_success', 'autosharp_success']]
X_dials = full_dials_df.iloc[:, :-4]

X_dials_train, X_dials_test, y_dials_train, y_dials_test = train_test_split(X_dials, y_dials,
                                                                        test_size=0.2,
                                                                        random_state=42,
                                                                        stratify=y_dials)
print("Training X shape: ", X_dials_train.shape)
print("Testing X shape: ", X_dials_test.shape)
print("Training y shape: ", y_dials_train.shape)
print("Training y shape: ", y_dials_test.shape)
print(X_dials_train[:10], y_dials_train[:10])




def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(10, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=200)
        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        print('>%.3f' % acc)
        results.append(acc)
    return results


# evaluate model
results = evaluate_model(X_3dii.to_numpy(), y_3dii.to_numpy())
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))





