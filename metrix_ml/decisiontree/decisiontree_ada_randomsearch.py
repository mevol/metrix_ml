###############################################################################
#
#  imports and set up environment
#
###############################################################################
'''Defining the environment for this class'''
import argparse
import pandas as pd
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import seaborn as sns
#import scikitplot as skplt
import imblearn
import joblib
import logging
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
#from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import resample
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from datetime import datetime
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.ensemble import AdaBoostClassifier
from tbx import get_confidence_interval
from tbx import feature_importances_best_estimator
from tbx import feature_importances_error_bars
from tbx import confusion_matrix_and_stats
from tbx import training_cv_stats
from tbx import testing_predict_stats
from tbx import plot_hist_pred_proba
from tbx import plot_precision_recall_vs_threshold
from tbx import plot_roc_curve

def make_output_folder(outdir):
  output_dir = os.path.join(outdir, 'decisiontree_ada_randomsearch')
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

class TreeAdaBoostRandSearch():
    '''This class is the doing the actual work in the following steps:
       * define smaller data frames: database, man_add, transform
       * split the data into training and test set
       * setup and run a randomized search for best paramaters to define a random forest
       * create a new random forest with best parameters
       * predict on this new random forest with test data and cross-validated training data
       * analyse the predisctions with graphs and stats
    '''
    def __init__(self, data, directory, numf, numc, bootiter):
        self.numf = numf
        self.numc = numc
        self.bootiter = bootiter
        self.data = pd.read_csv(data)
        self.directory = make_output_folder(directory)

        logging.basicConfig(level=logging.INFO, filename=os.path.join(self.directory,
                    'decisiontree_ada_randomsearch.log'), filemode='w')
        logging.info(f'Loaded input data')
        logging.info(f'Created output directories at {self.directory}')

        self.prepare_data()
        self.randomised_search()
        self.get_training_testing_prediction_stats()
        self.detailed_analysis()
        
    def prepare_data(self):
        print('*' *80)
        print('*    Preparing input data and split in train/test/calibration set')
        print('*' *80)

        for name in self.data.columns:
            if 'success' in name:
                y = self.data[name]
                X = self.data.drop([name, 'Unnamed: 0'], axis=1).select_dtypes(
                                                 exclude=['object'])

        # create a 5% calibration set if needed
        X_temp, self.X_cal, y_temp, self.y_cal = train_test_split(X, y, test_size=0.05,
                                                        random_state=42)

        # use the remaining data for 80/20 train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_temp,
                                                                        y_temp,
                                                                        test_size=0.2,
                                                                        random_state=100)

        logging.info(f'Created test, train and validation set')

    def randomised_search(self):

        print('*' *80)
        print('*    Running randomized search to find best classifier')
        print('*' *80)

        #create the decision forest
        clf1 = DecisionTreeClassifier(random_state=20,
                                      class_weight='balanced',
                                      max_features = self.numf)

        ada = AdaBoostClassifier(base_estimator=clf1,
                                 algorithm ="SAMME.R",
                                 random_state=55)

        logging.info(f'Initialised decision tree and AdaBoost using balanced class weights')

        #set up randomized search
        param_dict = {
                   'base_estimator__criterion': ['gini', 'entropy'],
                   'n_estimators': randint(100, 10000),#number of base estimators to use
                   'learning_rate': uniform(0.0001, 1.0),
                   'base_estimator__min_samples_split': randint(2, 20),
                   'base_estimator__max_depth': randint(1, 10),
                   'base_estimator__min_samples_leaf': randint(1, 20),
                   'base_estimator__max_leaf_nodes': randint(10, 20)}

        logging.info(f'Following parameters will be explored in randomized search \n'
                     f'{param_dict}')

        #building and running the randomized search
        rand_search = RandomizedSearchCV(ada, param_dict, random_state=5,
                                         cv=3, n_iter=self.numc,
                                         scoring='accuracy', n_jobs=-1)

        rand_search_fitted = rand_search.fit(self.X_train,
                                             self.y_train)
                                             
        best_parameters = rand_search_fitted.best_params_
        best_scores = rand_search_fitted.best_score_

        logging.info(f'Running randomised search for best patameters of a decision tree \n'
                     f'with AdaBoost scoring is accuracy \n'
                     f'Best parameters found: {best_parameters} \n'
                     f'Best accuracy scores found: {best_scores} \n')
                     
        self.model = rand_search_fitted.best_estimator_

        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        joblib.dump(self.model, os.path.join(self.directory,
                    'best_forest_rand_ada_'+datestring+'.pkl'))

        logging.info(f'Writing best classifier to disk in {self.directory} \n')

        print('*' *80)
        print('*    Getting 95%% confidence interval for best classifier')
        print('*' *80)

        alpha, upper, lower = get_confidence_interval(self.X_train, self.y_train,
                                                      self.X_test, self.y_test,
                                                      self.model, self.directory,
                                                      self.bootiter)

        logging.info(f'{alpha}% confidence interval {upper}% and {lower}% \n')
                                                      
        print('*' *80)
        print('*    Getting feature importances for best classifier')
        print('*' *80)


        best_clf_feat_import = self.model.feature_importances_
        best_clf_feat_import_sorted = sorted(zip(best_clf_feat_import,
                                                self.X_train.columns),
                                                reverse=True)

        logging.info(f'Feature importances for best classifier {best_clf_feat_import_sorted} \n')

        all_clf_feat_import_mean = np.mean(
                 [tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        all_clf_feat_import_mean_sorted = sorted(zip(all_clf_feat_import_mean,
                                                self.X_train.columns),
                                                reverse=True)

        print('*' *80)
        print('*    Plotting feature importances across all trees')
        print('*' *80)
        
        feature_importances_best_estimator(best_clf_feat_import_sorted, self.directory)
        logging.info(f'Plotting feature importances for best classifier in decreasing order \n')
        feature_importances_error_bars(self.model, self.X_train.columns, self.directory)
        logging.info(f'Plotting feature importances for best classifier with errorbars \n')

    def get_training_testing_prediction_stats(self):
        print('*' *80)
        print('*    Getting basic stats for training set and cross-validation')
        print('*' *80)

        training_stats, y_train_pred, y_train_pred_proba = training_cv_stats(
                                                self.model, self.X_train, self.y_train)
        print(training_stats)

        print(training_stats['acc'])
        logging.info(f'Basic stats achieved for training set and 3-fold CV \n'
            f'Accuracy for each individual fold of 3 CV folds: {training_stats["acc_cv"]} \n'
            f'Accuracy across all 3 CV-folds: {training_stats["acc"]} \n'
            f'ROC_AUC across all 3 CV-folds: {training_stats["roc_auc"]} \n'
            f'Recall across all 3 CV-folds: {training_stats["recall"]} \n'
            f'Precision across all 3 CV-folds: {training_stats["precision"]} \n'
            f'F1 score across all 3 CV-folds: {training_stats["f1-score"]} \n'
            f'Storing cross-validated y_train classes in y_train_pred \n'
            f'Storing cross-validated y_train probabilities in y_train_pred_proba \n')

        print('*' *80)
        print('*    Getting class predictions and probabilities for test set')
        print('*' *80)

        test_stats, self.y_pred, self.y_pred_proba = testing_predict_stats(
                                                self.model, self.X_test, self.y_test)

        logging.info(f'Predicting on the test set. \n'
                     f'Storing classes in y_pred and probabilities in y_pred_proba \n')

        print('*' *80)
        print('*    Calculate prediction stats for y_pred and y_pred_proba of test set')
        print('*' *80)

        logging.info(f'Basic stats on the test set. \n'
                     f'Prediction accuracy on the test set: {test_stats["predict_acc"]} \n'
                     f'Class distributio in the test set: {test_stats["class_distribution"]} \n'
                     f'Matthews Correlation Coefficient: {test_stats["mcc"]} \n'
                     f'Average number of class 1 samples: {test_stats["class_one"]} \n'
                     f'Average number of class 0 samples: {test_stats["class_zero"]} \n'
                     f'Null accuracy: {test_stats["null_acc"]} \n')

        print('*' *80)
        print('*    Plotting histogram for class 1 prediction probabilities for test set')
        print('*' *80)

        #store the predicted probabilities for class 1 of test set
        self.y_pred_proba_ones = self.y_pred_proba[:, 1]


        plot_hist_pred_proba(self.y_pred_proba_ones, self.directory)

        logging.info(
          f'Plotting prediction probabilities for class 1 in test set in histogram. \n')

    def detailed_analysis(self):
        print('*' *80)
        print('*    Making a confusion matrix for test set classification outcomes')
        print('*' *80)


        matrix_stats = confusion_matrix_and_stats(self.y_test, self.y_pred,
                                                  self.directory)

        logging.info(f'Detailed analysis of confusion matrix for test set. \n'
                     f'True positives: {matrix_stats["TP"]} \n'
                     f'True negatives: {matrix_stats["TN"]} \n'
                     f'False positives: {matrix_stats["FP"]} \n'
                     f'False negatives: {matrix_stats["FN"]} \n'
                     f'Classification accuracy: {matrix_stats["acc"]} \n'
                     f'Classification error: {matrix_stats["err"]} \n'
                     f'Sensitivity: {matrix_stats["sensitivity"]} \n'
                     f'Specificity: {matrix_stats["specificity"]} \n'
                     f'False positive rate: {matrix_stats["FP-rate"]} \n'
                     f'False negative rate: {matrix_stats["FN-rate"]} \n'
                     f'Precision: {matrix_stats["precision"]} \n'
                     f'F1-score: {matrix_stats["F1-score"]} \n')

        print('*' *80)
        print('*    Plotting precision recall curve for test set class 1 probabilities')
        print('*' *80)

        logging.info(
          f'Plotting precision recall curve for class 1 in test set probabilities. \n')
        
        plot_precision_recall_vs_threshold(self.y_test, self.y_pred_proba_ones,
                                           self.directory)

        print('*' *80)
        print('*    Plotting ROC curve ad calculating AUC for test set class 1 probabilities')
        print('*' *80)

        logging.info(
          f'Plotting ROC curve for class 1 in test set probabilities. \n')
        
        plot_roc_curve(self.y_test, self.y_pred_proba_ones, self.directory)

        AUC = round(roc_auc_score(self.y_test, self.y_pred_proba_ones) * 100, 2)

        logging.info(
          f'Calculating AUC for ROC curve for class 1 in test set probabilities: {AUC} \n')


def run(input_csv, output_dir, features, cycles, boot_iter):

    TreeAdaBoostRandSearch(input_csv, output_dir, features, cycles, boot_iter)



def main():
    '''defining the command line input to make it runable'''
    parser = argparse.ArgumentParser(description='AdaBoost and DecisionTree randomized search')

    parser.add_argument(
        '--input', 
        type=str, 
        dest='input',
        default='',
        help='The input CSV file')

    parser.add_argument(
        '--outdir',
        type=str,
        dest='outdir',
        default='',
        help='Specify output directory')
      
    parser.add_argument(
        '--num_features',
        type=int,
        dest='num_features',
        default=10,
        help='Number of features to look for')
      
    parser.add_argument(
        '--num_cycles',
        type=int,
        dest='num_cycles',
        default=500,
        help='Number of randomized search cycles')
      
    parser.add_argument(
        '--boot_iter',
        type=int,
        dest='boot_iter',
        default=1000,
        help='Number of bootstrap cycles')


    args = parser.parse_args()
    
    if args.input == '':
        parser.print_help()
        exit(0)
    
    run(args.input,
        args.outdir,
        args.num_features,
        args.num_cycles,
        args.boot_iter)


if __name__ == "__main__":
    main()
