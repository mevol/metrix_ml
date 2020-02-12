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
import scikitplot as skplt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.tree import export_graphviz
from datetime import datetime
from sklearn.externals import joblib
from scipy.stats import randint

###############################################################################
#
#  define command line arguments
#
###############################################################################

def parse_command_line():
  '''defining the command line input to make it runable'''
  parser = argparse.ArgumentParser(description='Random forest randomized search')
  
  parser.add_argument(
    '--input', 
    type=str, 
    dest="input",
    default="",
    help='The input CSV file')
    
  parser.add_argument(
    '--outdir',
    type=str,
    dest='outdir',
    default='',
    help='Specify output directory')

  args = parser.parse_args()
  if args.input == '':
    parser.print_help()
    exit(0)
  return args

###############################################################################
#
#  load the data from CSV file and creating output directory
#
###############################################################################

def load_metrix_data(csv_path):
  '''load the raw data as stored in CSV file'''
  return pd.read_csv(csv_path)

def make_output_folder(outdir):
  output_dir = os.path.join(outdir, 'decisiontree_randomsearch')
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

###############################################################################
#
#  class for ML using random forest with randomised search
#
###############################################################################

class DecisionTreeRandSearch(object):
  '''This class is the doing the actual work in the following steps:
     * define smaller data frames: database, man_add, transform
     * split the data into training and test set
     * setup and run a grid search for best paramaters to define a decsion tree
     * create a new tree with best parameters
     * predict on this new tree with test data and cross-validated training data
     * analyse the predisctions with graphs and stats
  '''
  def __init__(self, metrix, output_dir): 
    self.metrix = metrix
    self.output_dir = output_dir
    self.prepare_metrix_data()
    self.split_data()
    self.rand_search()
    self.tree_best_params()
    self.predict()
    self.analysis()
 
###############################################################################
#
#  creating smaller data frame
#
###############################################################################

  def prepare_metrix_data(self):
    '''Function to create smaller dataframe with columns of interest.
    ******
    Input: large data frame
    Output: smaller dataframe
    '''
    print('*' *80)
    print('*    Preparing input dataframe X_metrix')
    print('*' *80)

    #database plus manually added data
    self.X_metrix = self.metrix[['eLLG', 'seq_ident', 'MW_chain']]

    self.X_metrix = self.X_metrix.fillna(0)

    with open(os.path.join(self.output_dir,
              'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created dataframe X_metrix \n')
      text_file.write('with columns: \n')
      text_file.write(str(self.X_metrix.columns)+ '\n')

###############################################################################
#
#  creating training and test set 
#
###############################################################################

  def split_data(self):
    '''Function which splits the input data into training set and test set.
    ******
    Input: a dataframe that contains the features and labels in columns and the samples
          in rows
    Output: sets of training and test data with an 80/20 split; X_train, X_test, y_train,
            y_test
    '''
    print('*' *80)
    print('*    Splitting data into test and training set with test=20%')
    print('*' *80)

    y = self.metrix['MR_success']

#stratified split of samples
    X_metrix_train, X_metrix_test, y_train, y_test = train_test_split(self.X_metrix, y, test_size=0.2, random_state=42, stratify=y)
    
    assert self.X_metrix.columns.all() == X_metrix_train.columns.all()

    self.X_metrix_train = X_metrix_train
    self.X_metrix_test = X_metrix_test
    self.y_train = y_train
    self.y_test = y_test

    with open(os.path.join(self.output_dir,
              'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('X_metrix: X_metrix_train, X_metrix_test \n')
      text_file.write('y(MR_success): y_train, y_test \n')

###############################################################################
#
#  randomized search for best parameter combination
#
###############################################################################

  def rand_search(self):
    '''running a randomized search to find the parameter combination for a decision tree
     which gives the best accuracy score'''
    print('*' *80)
    print('*    Running RandomizedSearch for best parameter combination for DecisionTree')
    print('*' *80)

    #create the decision tree
    tree_clf_rand = DecisionTreeClassifier(random_state=100,
                                           class_weight='balanced')#class_weight='balanced'

    with open(os.path.join(self.output_dir,
              'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created decision tree: tree_clf_rand \n')

    #set up grid search
    param_rand = {"criterion": ["gini", "entropy"],#metric to judge reduction of impurity
                  #'max_features': 2,#max number of features when splitting
                  "min_samples_split": randint(2, 20),#min samples per node to induce split
                  "max_depth": randint(3, 10),#max number of splits to do
                  "min_samples_leaf": randint(1, 20),#min number of samples in a leaf; may set to 1 anyway
                  "max_leaf_nodes": randint(10, 20)}#max number of leaves

    with open(os.path.join(self.output_dir,
                           'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Running grid search for the following parameters: %s \n' %param_rand)
      text_file.write('use cv=3, scoring=accuracy \n')

    #building and running the grid search
    rand_search = RandomizedSearchCV(tree_clf_rand,
                                     param_rand,
                                     random_state=5,
                                     cv=3,
                                     n_iter=500,
                                     scoring='accuracy',
                                     n_jobs=-1)
                              
    rand_search_fitted = rand_search.fit(self.X_metrix_train, self.y_train)
    with open(os.path.join(self.output_dir,
                           'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Best parameters: ' +str(rand_search_fitted.best_params_)+'\n')
      text_file.write('Best score: ' +str(rand_search_fitted.best_score_)+'\n')
    feature_importances_fitted = rand_search_fitted.best_estimator_.feature_importances_
    feature_importances_fitted_ls = sorted(zip(feature_importances_fitted,
                                             self.X_metrix_train), reverse=True)
    with open(os.path.join(self.output_dir,
                           'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Feature importances: %s \n' %feature_importances_fitted_ls)
    
    self.best_params = rand_search_fitted.best_params_

###############################################################################
#
#  creating new tree with best parameter combination
#
###############################################################################

  def tree_best_params(self):
    '''create a new decision tree using the best parameter combination found
    above'''
    print('*' *80)
    print('*    Building new tree based on best parameter combination and save as pickle')
    print('*' *80)
    
    self.tree_clf_rand_new = DecisionTreeClassifier(**self.best_params,
                                                    random_state=100,
                                                    class_weight='balanced')#class_weight='balanced'
    self.tree_clf_rand_new.fit(self.X_metrix_train, self.y_train)

    def plot_features(clf, attr, directory):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      skplt.estimators.plot_feature_importances(clf,
                                                feature_names=attr,
                                                x_tick_rotation=90,
                                                max_num_features=len(attr),
                                                figsize=(25, 25),
                                                text_fontsize=14)
      plt.tight_layout()
      plt.savefig(os.path.join(directory,
                   'feature_importances_bar_plot_rand_tree_'+datestring+'.png'))
      plt.close()
   
    plot_features(self.tree_clf_rand_new,
                  self.X_metrix_train.columns,
                  self.output_dir)
   
    def write_pickle(tree, directory):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      joblib.dump(tree, os.path.join(directory,
                                     'best_tree_rand_'+datestring+'.pkl'))
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Created new decision tree "tree_clf_rand_new" using best parameters \n')
        text_file.write('Creating pickle file for best tree as best_tree_rand.pkl \n')
    
    write_pickle(self.tree_clf_rand_new,
                 self.output_dir)
    
    def visualise_tree(tree, directory, columns):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      dotfile = os.path.join(directory, 'tree_clf_rand_new_'+datestring+'.dot')
      pngfile = os.path.join(directory, 'tree_clf_rand_new_'+datestring+'.png')
      with open(dotfile, 'w') as f:
        export_graphviz(tree,
                        out_file=f,
                        feature_names=columns,
                        rounded=True,
                        filled=True)
                       
      command = ["dot", "-Tpng", dotfile, "-o", pngfile]
      subprocess.check_call(command)

      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Writing DOTfile and convert to PNG for "tree_clf_rand_new" \n')
        text_file.write('DOT filename: tree_clf_rand_new.dot \n')
        text_file.write('PNG filename: tree_clf_rand_new.png \n')
    
    visualise_tree(self.tree_clf_rand_new,
                   self.output_dir,
                   self.X_metrix_train.columns)

    print('*' *80)
    print('*    Getting basic stats for new tree')
    print('*' *80)

    def basic_stats(tree, X_train, directory):
      #distribution --> accuracy
      accuracy_each_cv = cross_val_score(tree,
                                         X_train,
                                         self.y_train,
                                         cv=3,
                                         scoring='accuracy')
      accuracy_mean_cv = cross_val_score(tree,
                                         X_train,
                                         self.y_train,
                                         cv=3,
                                         scoring='accuracy').mean()
      # calculate cross_val_scoring with different scoring functions for CV train set
      train_roc_auc = cross_val_score(tree,
                                      X_train,
                                      self.y_train,
                                      cv=3,
                                      scoring='roc_auc').mean()
      train_accuracy = cross_val_score(tree,
                                       X_train,
                                       self.y_train,
                                       cv=3,
                                       scoring='accuracy').mean()
      train_recall = cross_val_score(tree,
                                     X_train,
                                     self.y_train,
                                     cv=3,
                                     scoring='recall').mean()
      train_precision = cross_val_score(tree,
                                        X_train,
                                        self.y_train,
                                        cv=3,
                                        scoring='precision').mean()
      train_f1 = cross_val_score(tree,
                                 X_train,
                                 self.y_train,
                                 cv=3,
                                 scoring='f1').mean()

      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Get various cross_val_scores to evaluate clf performance for best parameters \n')     
        text_file.write('Accuracy for each of 3 CV folds: %s \n' %accuracy_each_cv)
        text_file.write('Mean accuracy over all 3 CV folds: %s \n' %accuracy_mean_cv)
        text_file.write('ROC_AUC mean for 3-fold CV: %s \n' %train_roc_auc)
        text_file.write('Accuracy mean for 3-fold CV: %s \n' %train_accuracy)
        text_file.write('Recall mean for 3-fold CV: %s \n' %train_recall)
        text_file.write('Precision mean for 3-fold CV: %s \n' %train_precision)
        text_file.write('F1 score mean for 3-fold CV: %s \n' %train_f1)
    
    basic_stats(self.tree_clf_rand_new,
                self.X_metrix_train,
                self.output_dir)

###############################################################################
#
#  Predicting with test set and cross-validation set using the bets forest
#
###############################################################################

  def predict(self):
    '''do predictions using the best tree an the test set as well as training
       set with 3 cross-validation folds and doing some initial analysis on the
       output'''
    print('*' *80)
    print('*    Predict using new tree and test/train_CV set')
    print('*' *80)

    #try out how well the classifier works to predict from the test set
    self.y_pred = self.tree_clf_rand_new.predict(self.X_metrix_test)
    self.y_pred_proba = self.tree_clf_rand_new.predict_proba(self.X_metrix_test)
    with open(os.path.join(self.output_dir,
              'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions and probabilities for X_metrix_test in y_pred and probabilities y_pred_proba\n')

    #alternative way to not have to use the test set
    self.y_train_CV_pred = cross_val_predict(self.tree_clf_rand_new,
                                             self.X_metrix_train,
                                             self.y_train,
                                             cv=3)
    self.y_train_CV_pred_proba = cross_val_predict(self.tree_clf_rand_new,
                                                   self.X_metrix_train,
                                                   self.y_train,
                                                   cv=3,
                                                   method='predict_proba')
    with open(os.path.join(self.output_dir,
              'decisiontree_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions and probabilities for X_metrix_train with 3-fold CV in y_train_pred \n')

    print('*' *80)
    print('*    Calculate prediction stats')
    print('*' *80)

    def prediction_stats(y_test, y_pred, directory):
      # calculate accuracy
      y_accuracy = metrics.accuracy_score(self.y_test, y_pred)

      # examine the class distribution of the testing set (using a Pandas Series method)
      class_dist = self.y_test.value_counts()

      # calculate the percentage of ones
      # because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
      ones = self.y_test.mean()

      # calculate the percentage of zeros
      zeros = 1 - self.y_test.mean()

      # calculate null accuracy in a single line of code
      # only for binary classification problems coded as 0/1
      null_acc = max(self.y_test.mean(), 1 - self.y_test.mean())

      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Accuracy score or agreement between y_test and y_pred_class: %s \n' %y_accuracy)
        text_file.write('Class distribution for y_test: %s \n' %class_dist)
        text_file.write('Percent 1s in y_test: %s \n' %ones)
        text_file.write('Percent 0s in y_test: %s \n' %zeros)
        text_file.write('Null accuracy in y_test: %s \n' %null_acc)
    
    prediction_stats(self.y_test,
                     self.y_pred,
                     self.output_dir)
    
###############################################################################
#
#  detailed analysis and stats
#
###############################################################################

  def analysis(self):
    '''detailed analysis of the output:
       * create a confusion matrix
       * split the data into TP, TN, FP, FN for test and train_CV
       * determine accuracy score
       * determine classification error
       * determine sensitivity
       * determine specificity
       * determine false-positive rate
       * determine precision
       * determine F1 score
       calculate prediction probabilities and draw plots
       * histogram for probability to be class 1
       * precision-recall curve
       * look for adjustments in classification thresholds
       * ROC curve
       * determine ROC_AUC
       * try different scoring functions for comparison'''
    print('*' *80)
    print('*    Detailed analysis and plotting')
    print('*' *80)
    
    def conf_mat(y_test, y_train, y_pred, y_train_pred, directory):
      # IMPORTANT: first argument is true values, second argument is predicted values
      # this produces a 2x2 numpy array (matrix)
      conf_mat_test = metrics.confusion_matrix(y_test, y_pred)
      conf_mat_3CV = metrics.confusion_matrix(y_train, y_train_pred)
      def draw_conf_mat(matrix, directory):
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        labels = ['0', '1']      
        ax = plt.subplot()
        sns.heatmap(matrix, annot=True, ax=ax)
        plt.title('Confusion matrix of the classifier')
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(directory,
                               'confusion_matrix_tree_rand_'+datestring+'.png'))
        plt.close()

      draw_conf_mat(conf_mat_test, directory)
      #draw_conf_mat(conf_mat_3CV, directory, 'train_CV_')
      
      TP = conf_mat_test[1, 1]
      TN = conf_mat_test[0, 0]
      FP = conf_mat_test[0, 1]
      FN = conf_mat_test[1, 0]
      
      TP_CV = conf_mat_3CV[1, 1]
      TN_CV = conf_mat_3CV[0, 0]
      FP_CV = conf_mat_3CV[0, 1]
      FN_CV = conf_mat_3CV[1, 0]

      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('confusion matrix using test set: %s \n' %conf_mat_test)
        text_file.write('confusion matrix using 3-fold CV: %s \n' %conf_mat_3CV)
        text_file.write('Slicing confusion matrix for test set into: TP, TN, FP, FN \n')
        text_file.write('Slicing confusion matrix for 3-fold CV into: TP_CV, TN_CV, FP_CV, FN_CV \n')
      
      #calculate accuracy
      acc_score_man_test = (TP + TN) / float(TP + TN + FP + FN)
      acc_score_sklearn_test = metrics.accuracy_score(y_test, y_pred)
      acc_score_man_CV = (TP_CV + TN_CV) / float(TP_CV + TN_CV + FP_CV + FN_CV)
      acc_score_sklearn_CV = metrics.accuracy_score(y_train, y_train_pred)  
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Accuracy score: \n')
        text_file.write('accuracy score manual test: %s \n' %acc_score_man_test)
        text_file.write('accuracy score sklearn test: %s \n' %acc_score_sklearn_test)
        text_file.write('accuracy score manual CV: %s \n' %acc_score_man_CV)
        text_file.write('accuracy score sklearn CV: %s \n' %acc_score_sklearn_CV)
        
      #classification error
      class_err_man_test = (FP + FN) / float(TP + TN + FP + FN)
      class_err_sklearn_test = 1 - metrics.accuracy_score(y_test, y_pred)
      class_err_man_CV = (FP_CV + FN_CV) / float(TP_CV + TN_CV + FP_CV + FN_CV)
      class_err_sklearn_CV = 1 - metrics.accuracy_score(y_train, y_train_pred)
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Classification error: \n')  
        text_file.write('classification error manual test: %s \n' %class_err_man_test)
        text_file.write('classification error sklearn test: %s \n' %class_err_sklearn_test)
        text_file.write('classification error manual CV: %s \n' %class_err_man_CV)
        text_file.write('classification error sklearn CV: %s \n' %class_err_sklearn_CV)
        
      #sensitivity/recall/true positive rate; correctly placed positive cases  
      sensitivity_man_test = TP / float(FN + TP)
      sensitivity_sklearn_test = metrics.recall_score(y_test, y_pred)
      sensitivity_man_CV = TP_CV / float(FN_CV + TP_CV)
      sensitivity_sklearn_CV = metrics.recall_score(y_train, y_train_pred)
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Sensitivity/Recall/True positives: \n')
        text_file.write('sensitivity manual test: %s \n' %sensitivity_man_test)
        text_file.write('sensitivity sklearn test: %s \n' %sensitivity_sklearn_test)
        text_file.write('sensitivity manual CV: %s \n' %sensitivity_man_CV)
        text_file.write('sensitivity sklearn CV: %s \n' %sensitivity_sklearn_CV)
      
      #specificity  
      specificity_man_test = TN / (TN + FP)
      specificity_man_CV = TN_CV / (TN_CV + FP_CV)
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Specificity: \n')
        text_file.write('specificity manual test: %s \n' %specificity_man_test)
        text_file.write('specificity manual CV: %s \n' %specificity_man_CV)
      
      #false positive rate  
      false_positive_rate_man_test = FP / float(TN + FP)
      false_positive_rate_man_CV = FP_CV / float(TN_CV + FP_CV)
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('False positive rate or 1-specificity: \n')
        text_file.write('false positive rate manual test: %s \n' %false_positive_rate_man_test)
        text_file.write('1 - specificity test: %s \n' %(1 - specificity_man_test))
        text_file.write('false positive rate manual CV: %s \n' %false_positive_rate_man_CV)
        text_file.write('1 - specificity CV: %s \n' %(1 - specificity_man_CV))
      
      #precision/confidence of placement  
      precision_man_test = TP / float(TP + FP)
      precision_sklearn_test = metrics.precision_score(y_test, y_pred)
      precision_man_CV = TP_CV / float(TP_CV + FP_CV)
      precision_sklearn_CV = metrics.precision_score(y_train, y_train_pred)
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Precision or confidence of classification: \n')
        text_file.write('precision manual: %s \n' %precision_man_test)
        text_file.write('precision sklearn: %s \n' %precision_sklearn_test)
        text_file.write('precision manual CV: %s \n' %precision_man_CV)
        text_file.write('precision sklearn CV: %s \n' %precision_sklearn_CV)
      
      #F1 score; uses precision and recall  
      f1_score_sklearn_test = f1_score(y_test, y_pred)
      f1_score_sklearn_CV = f1_score(y_train, y_train_pred)
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('F1 score: \n')
        text_file.write('F1 score sklearn test: %s \n' %f1_score_sklearn_test)
        text_file.write('F1 score sklearn CV: %s \n' %f1_score_sklearn_CV)
        
    conf_mat(self.y_test,
             self.y_train,
             self.y_pred,
             self.y_train_CV_pred,
             self.output_dir)

    def prediction_probas(tree, X_train, y_train, X_test, y_test, y_pred_proba,
                                              y_train_CV_pred_proba, directory): 
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')      
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Plotting histogram for y_pred_proba_train_CV \n')
        text_file.write('Plotting histogram for y_pred_proba_test \n')
   
      #plot histograms of probabilities  
      def plot_hist_pred_proba(y_pred_proba, directory):
        plt.hist(y_pred_proba, bins=20)
        plt.xlim(0,1)
        plt.title('Histogram of predicted probabilities for y_pred_proba to be class 1')
        plt.xlabel('Predicted probability of EP_success')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(directory,
                                'hist_pred_proba_tree_rand_'+datestring+'.png'))
        plt.close()

      #plot_hist_pred_proba(y_train_CV_pred_proba[:, 1], 'train_CV_', directory)
      plot_hist_pred_proba(y_pred_proba[:, 1], directory)
      
      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Getting y_scores for y_pred_proba_train_CV and y_pred_proba_test as y_scores_train_CV and y_scores_test for class 0 and 1\n')

      self.y_scores_ones = y_pred_proba[:, 1]#test data to be class 1
      self.y_scores_CV_ones = y_train_CV_pred_proba[:, 1]#training data to be class 1

      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Plotting Precision-Recall for y_test and y_scores_test \n')
        text_file.write('Plotting Precision-Recall for y_train and y_scores_train_CV \n')
      
      #plot precision and recall curve
      def plot_precision_recall_vs_threshold(precisions,
                                             recalls,
                                             thresholds_tree,
                                             classes,
                                             directory):
        plt.plot(thresholds_tree, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds_tree, recalls[:-1], "g--", label="Recall")
        plt.title('Precsion-Recall plot for for MR_success classifier for class %s' %(classes))
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0,1])
        plt.savefig(os.path.join(directory,
                    'Precision_Recall_tree_rand_'+datestring+classes+'.png'))
        plt.close()

     #plot Precision Recall Threshold curve for test set        
      precisions, recalls, thresholds_tree = precision_recall_curve(
                                                             self.y_test,
                                                             self.y_scores_ones)
      plot_precision_recall_vs_threshold(precisions,
                                         recalls,
                                         thresholds_tree,
                                         '1',
                                         directory)
#      #plot Precision Recall Threshold curve for CV train set       
#      precisions, recalls, thresholds_tree = precision_recall_curve(self.y_train, self.y_scores_CV_ones)
#      plot_precision_recall_vs_threshold(precisions, recalls, thresholds_tree, 'train_CV_', '1', directory)
           
      with open(os.path.join(directory, 'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('Plotting ROC curve for y_test and y_scores_test \n')
        text_file.write('Plotting ROC curve for y_train and y_scores_train_CV \n')

      #IMPORTANT: first argument is true values, second argument is predicted probabilities
      #we pass y_test and y_pred_prob
      #we do not use y_pred_class, because it will give incorrect results without generating an error
      #roc_curve returns 3 objects fpr, tpr, thresholds
      #fpr: false positive rate
      #tpr: true positive rate

      #plot ROC curves
      def plot_roc_curve(y_test, y_proba, directory):
        skplt.metrics.plot_roc(y_test, y_proba, title='ROC curve')
        plt.savefig(os.path.join(directory,
                                'ROC_curve_skplt_tree_rand_'+datestring+'.png'))
        plt.close()
        
#      plot_roc_curve(self.y_train, y_train_CV_pred_proba, 'train_CV_', directory)  
      plot_roc_curve(self.y_test,
                     y_pred_proba,
                     directory)  
    
      def plot_roc_curve(fpr, tpr, classes, directory):
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.title('ROC curve for MR_success classifier for class %s' %(classes)) 
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        plt.savefig(os.path.join(directory,
                              'ROC_curve_tree_rand_'+datestring+classes+'.png'))
        plt.close()
        
      #ROC curve for test set      
      fpr_1, tpr_1, thresholds_1 = roc_curve(self.y_test, self.y_scores_ones)
      plot_roc_curve(fpr_1, tpr_1, '1', directory)
#      #ROC curve for 10-fold CV train set      
#      fpr_CV_1, tpr_CV_1, thresholds_CV_1 = roc_curve(self.y_train, self.y_scores_CV_ones)
#      plot_roc_curve(fpr_CV_1, tpr_CV_1, 'train_CV_', '1', directory)
      
      #calculate the area under the curve to get the performance for a classifier
      # IMPORTANT: first argument is true values, second argument is predicted probabilities
      AUC_test_class1 = metrics.roc_auc_score(self.y_test, self.y_scores_ones)
#      AUC_train_class1 = metrics.roc_auc_score(self.y_train, self.y_scores_CV_ones)

      with open(os.path.join(directory,
                'decisiontree_randomsearch.txt'), 'a') as text_file:
        text_file.write('AUC for test set class 1: %s \n' %AUC_test_class1)
#        text_file.write('AUC for CV train set class 1: %s \n' %AUC_train_class1)

    prediction_probas(self.tree_clf_rand_new,
                      self.X_metrix_train,
                      self.y_train,
                      self.X_metrix_test,
                      self.y_test,
                      self.y_pred_proba,
                      self.y_train_CV_pred_proba,
                      self.output_dir)    
 
def run():
  args = parse_command_line()
  
  
###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  output_dir = make_output_folder(args.outdir)

###############################################################################

  decision_tree_rand_search = DecisionTreeRandSearch(metrix, output_dir)

