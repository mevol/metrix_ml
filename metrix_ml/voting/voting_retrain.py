###############################################################################
#
#  imports and set up environment
#
###############################################################################
'''Defining the environment for this class'''
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import seaborn as sns
import scikitplot as skplt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from datetime import datetime
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

###############################################################################
#
#  define command line arguments
#
###############################################################################

def parse_command_line():
  '''defining the command line input to make it runable'''
  parser = argparse.ArgumentParser(description='Random forest grid search')
  
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
#  load the data from CSV file
#
###############################################################################

def load_metrix_data(csv_path):
  '''load the raw data as stored in CSV file'''
  return pd.read_csv(csv_path)

def make_output_folder(outdir):
  names = ['newdata_minusEP', 'bbbb']
  result = []
  for name in names:
    name = os.path.join(outdir, 'voting', name)
    os.makedirs(name, exist_ok=True)
    result.append(name)
  return result

###############################################################################
#
#  class for ML using random forest with randomised search
#
###############################################################################

class Ensemble(object):
  '''This class is the doing the actual work in the following steps:
     * define smaller data frames: database, man_add, transform
     * split the data into training and test set
     * setup and run a grid search for best paramaters to define a random forest
     * create a new random forest with best parameters
     * predict on this new random forest with test data and cross-validated training data
     * analyse the predisctions with graphs and stats
  '''
  def __init__(self, metrix, newdata_minusEP, bbbb):
    self.metrix=metrix
    self.newdata_minusEP=newdata_minusEP
    #self.prepare_metrix_data()
    self.split_data()
    self.load_models()
    self.voting()
    self.predict()
    self.analysis()

    ###############################################################################
    #
    #  creating training and test set for each of the 3 dataframes
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

    self.X_metrix = self.metrix[['diffI', 'anomalousCC', 'lowreslimit', 'anomalousslope', 'diffF']]

    y = self.metrix['EP_success']

    X_newdata_transform_train, X_newdata_transform_test, y_train, y_test = train_test_split(self.X_metrix, y, test_size=0.2, random_state=42, stratify=y)
    
    #assert self.X_newdata_transform.columns.all() == X_newdata_transform_train.columns.all()

    self.X_newdata_transform_train = X_newdata_transform_train
    self.X_newdata_transform_test = X_newdata_transform_test
    self.y_train = y_train
    self.y_test = y_test

    with open(os.path.join(self.newdata_minusEP, 'voting.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_transform: X_transform_train, X_transform_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')

    ###############################################################################
    #
    #  random search for best parameter combination
    #
    ###############################################################################

  def load_models(self):
    '''running a randomized search to find the parameter combination for a random forest
     which gives the best accuracy score'''
    print('*' *80)
    print('*    Creating models equivalent to Decisiontree_ada, Decisiontree_bag and Randomforest')
    print('*' *80)

#    forest = joblib.load('/dls/mx-scratch/melanie/for_METRIX/test_metrix_ml/20181128/results/calibrate/randomforest_randomsearch/calibrate/calibrate/calibrated_classifier_20190226_1036.pkl')

    self.forest = RandomForestClassifier(criterion='gini',
                                         max_depth=7,
                                         max_features=2,
                                         max_leaf_nodes=19,
                                         min_samples_leaf=1,
                                         min_samples_split=5,
                                         n_estimators=4169,
                                         random_state=42)
                                         
    self.forest.fit(self.X_newdata_transform_train, self.y_train)
    
#    tree_ada = joblib.load('/dls/mx-scratch/melanie/for_METRIX/test_metrix_ml/20181128/results/calibrate/decisiontree_ada/calibrate/calibrate/calibrated_classifier_20190226_1157.pkl')
    
    clf1 = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=9,
                                  max_features=3,
                                  max_leaf_nodes=14,
                                  min_samples_leaf=1,
                                  min_samples_split=9,
                                  random_state= 0)
    self.tree_ada = AdaBoostClassifier(clf1,
                                       learning_rate=0.8404169268999371,
                                       n_estimators=2832,
                                       algorithm ="SAMME.R",
                                       random_state=5)
                                
    self.tree_ada.fit(self.X_newdata_transform_train, self.y_train)
   
#    tree_bag = joblib.load('/dls/mx-scratch/melanie/for_METRIX/test_metrix_ml/20181128/results/calibrate/decisiontree_bag/calibrate/calibrate/calibrated_classifier_20190226_1209.pkl')

    clf2 = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=6,
                                  max_features=3,
                                  max_leaf_nodes=17,
                                  min_samples_leaf=2,
                                  min_samples_split=2,
                                  random_state= 0)
    self.tree_bag = BaggingClassifier(clf2,
                                      n_estimators=6509,
                                      n_jobs=-1,
                                      bootstrap=True,
                                      random_state=100)
    self.tree_bag.fit(self.X_newdata_transform_train, self.y_train)   

    
    with open(os.path.join(self.newdata_minusEP, 'voting.txt'), 'a') as text_file:
      text_file.write('Creating models forest, tree_ada, tree_bag equivalent to saved ones \n')


    ###############################################################################
    #
    #  creating new forest with best parameter combination
    #
    ###############################################################################

  def voting(self):
    '''create a new random forest using the best parameter combination found above'''
    print('*' *80)
    print('*    Create voting classifier')
    print('*' *80)

    #create a dictionary of our models
    estimators=[('forest', self.forest),
                ('tree_ada', self.tree_ada),
                ('tree_bag', self.tree_bag)]

    #create our voting classifier, inputting our models
    voter = VotingClassifier(estimators, voting='soft')
    
    self.voter = voter.fit(self.X_newdata_transform_train, self.y_train)

    def write_pickle(clf, directory):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      joblib.dump(clf, os.path.join(directory,'best_voter_'+datestring+'.pkl'))
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Created new voting classifier \n')
        text_file.write('Creating pickle file for voting classifier \n')
    
    write_pickle(self.voter, self.newdata_minusEP)

    ###############################################################################
    #
    #  Predicting with test set and cross-validation set using the bets forest
    #
    ###############################################################################

  def predict(self):
    '''do predictions using the best random forest an the test set as well as training set with
       10 cross-validation folds and doing some initial analysis on the output'''
    print('*' *80)
    print('*    Predict using new forest and test/train_CV set')
    print('*' *80)

    self.y_pred_voter = self.voter.predict(self.X_newdata_transform_test)
    self.y_pred_proba_voter = self.voter.predict_proba(self.X_newdata_transform_test)
    with open(os.path.join(self.newdata_minusEP, 'voting.txt'), 'a') as text_file:
      text_file.write('Saving predictions and probabilities for X_transform_test in y_pred_voter and probabilities y_pred_proba_voter\n')

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

      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Accuracy score or agreement between y_test and y_pred: %s \n' %y_accuracy)
        text_file.write('Class distribution for y_test: %s \n' %class_dist)
        text_file.write('Percent 1s in y_test: %s \n' %ones)
        text_file.write('Percent 0s in y_test: %s \n' %zeros)
        text_file.write('Null accuracy in y_test: %s \n' %null_acc)
    
    prediction_stats(self.y_test, self.y_pred_voter, self.newdata_minusEP)

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

    def conf_mat(y_test,  y_pred, directory):
      # IMPORTANT: first argument is true values, second argument is predicted values
      # this produces a 2x2 numpy array (matrix)
      conf_mat_test = metrics.confusion_matrix(y_test, y_pred)
      def draw_conf_mat(matrix, directory, name):
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        labels = ['0', '1']      
        ax = plt.subplot()
        sns.heatmap(matrix, annot=True, ax=ax)
        plt.title('Confusion matrix of the classifier')
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(directory, 'confusion_matrix_voter_'+name+datestring+'.png'))
        plt.close()

      draw_conf_mat(conf_mat_test, directory, 'test_')
      
      TP = conf_mat_test[1, 1]
      TN = conf_mat_test[0, 0]
      FP = conf_mat_test[0, 1]
      FN = conf_mat_test[1, 0]

      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('confusion matrix using test set: %s \n' %conf_mat_test)
        text_file.write('Slicing confusion matrix for test set into: TP, TN, FP, FN \n')
      
      #calculate accuracy
      acc_score_man_test = (TP + TN) / float(TP + TN + FP + FN)
      acc_score_sklearn_test = metrics.accuracy_score(y_test, y_pred)
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Accuracy score: \n')
        text_file.write('accuracy score manual test: %s \n' %acc_score_man_test)
        text_file.write('accuracy score sklearn test: %s \n' %acc_score_sklearn_test)
        
      #classification error
      class_err_man_test = (FP + FN) / float(TP + TN + FP + FN)
      class_err_sklearn_test = 1 - metrics.accuracy_score(y_test, y_pred)
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Classification error: \n')  
        text_file.write('classification error manual test: %s \n' %class_err_man_test)
        text_file.write('classification error sklearn test: %s \n' %class_err_sklearn_test)
        
      #sensitivity/recall/true positive rate; correctly placed positive cases  
      sensitivity_man_test = TP / float(FN + TP)
      sensitivity_sklearn_test = metrics.recall_score(y_test, y_pred)
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Sensitivity/Recall/True positives: \n')
        text_file.write('sensitivity manual test: %s \n' %sensitivity_man_test)
        text_file.write('sensitivity sklearn test: %s \n' %sensitivity_sklearn_test)
      
      #specificity  
      specificity_man_test = TN / (TN + FP)
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Specificity: \n')
        text_file.write('specificity manual test: %s \n' %specificity_man_test)
      
      #false positive rate  
      false_positive_rate_man_test = FP / float(TN + FP)
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('False positive rate or 1-specificity: \n')
        text_file.write('false positive rate manual test: %s \n' %false_positive_rate_man_test)
        text_file.write('1 - specificity test: %s \n' %(1 - specificity_man_test))
      
      #precision/confidence of placement  
      precision_man_test = TP / float(TP + FP)
      precision_sklearn_test = metrics.precision_score(y_test, y_pred)
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Precision or confidence of classification: \n')
        text_file.write('precision manual: %s \n' %precision_man_test)
        text_file.write('precision sklearn: %s \n' %precision_sklearn_test)
      
      #F1 score; uses precision and recall  
      f1_score_sklearn_test = f1_score(y_test, y_pred)
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('F1 score: \n')
        text_file.write('F1 score sklearn test: %s \n' %f1_score_sklearn_test)
        
    conf_mat(self.y_test, self.y_pred_voter, self.newdata_minusEP)
 
    def prediction_probas(tree, X_train, y_train, X_test, y_test, y_pred_proba, directory, kind): 
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')      
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Plotting histogram for y_pred_proba_test \n')
   
      #plot histograms of probabilities  
      def plot_hist_pred_proba(y_pred_proba, name, directory):
        plt.hist(y_pred_proba, bins=20)
        plt.xlim(0,1)
        plt.title('Histogram of predicted probabilities for y_pred_proba_%s to be class 1' %name)
        plt.xlabel('Predicted probability of EP_success')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(directory, 'hist_pred_proba_voter_'+name+datestring+'.png'))
        plt.close()

      plot_hist_pred_proba(y_pred_proba[:, 1], 'test_', directory)
      
      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Getting y_scores for y_pred_proba_test as y_scores_test for class 1\n')

      self.y_scores_ones = y_pred_proba[:, 1]#test data to be class 1

      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Plotting Precision-Recall for y_test and y_scores_test \n')
      
      #plot precision and recall curve
      def plot_precision_recall_vs_threshold(precisions, recalls, thresholds_tree, name, classes, directory):
        plt.plot(thresholds_tree, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds_tree, recalls[:-1], "g--", label="Recall")
        plt.title('Precsion-Recall plot for voting classifier using %s set to be class %s' %(name, classes))
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0,1])
        plt.savefig(os.path.join(directory, 'Precision_Recall_voter_'+name+datestring+classes+'.png'))
        plt.close()

     #plot Precision Recall Threshold curve for test set        
      precisions, recalls, thresholds_tree = precision_recall_curve(self.y_test, self.y_scores_ones)
      plot_precision_recall_vs_threshold(precisions, recalls, thresholds_tree, 'test_', '1', directory)

      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('Plotting ROC curve for y_test and y_scores_test \n')

      #IMPORTANT: first argument is true values, second argument is predicted probabilities
      #we pass y_test and y_pred_prob
      #we do not use y_pred, because it will give incorrect results without generating an error
      #roc_curve returns 3 objects fpr, tpr, thresholds
      #fpr: false positive rate
      #tpr: true positive rate
    
      #plot ROC curves
      def plot_roc_curve(y_test, y_proba, name, directory):
        skplt.metrics.plot_roc(y_test, y_proba, title='ROC curve %s' %name)
        plt.savefig(os.path.join(directory, 'ROC_curve_skplt_voter_'+name+datestring+'.png'))
        plt.close()
        
      plot_roc_curve(self.y_test, y_pred_proba, 'test_', directory)  
    
      def plot_roc_curve(fpr, tpr, name, classes, directory):
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.title('ROC curve for EP_success classifier using %s set for class %s' %(name, classes)) 
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'ROC_curve_voter_'+name+datestring+classes+'.png'))
        plt.close()
        
      #ROC curve for test set      
      fpr_1, tpr_1, thresholds_1 = roc_curve(self.y_test, self.y_scores_ones)
      plot_roc_curve(fpr_1, tpr_1, 'test_', '1', directory)
      
      #calculate the area under the curve to get the performance for a classifier
      # IMPORTANT: first argument is true values, second argument is predicted probabilities
      AUC_test_class1 = metrics.roc_auc_score(self.y_test, self.y_scores_ones)

      with open(os.path.join(directory, 'voting.txt'), 'a') as text_file:
        text_file.write('AUC for test set class 1: %s \n' %AUC_test_class1)

    prediction_probas(self.voter, self.X_newdata_transform_train, self.y_train, self.X_newdata_transform_test, self.y_test, self.y_pred_proba_voter, self.newdata_minusEP, 'newdata_minusEP')    

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  newdata_minusEP, bbbb= make_output_folder(args.outdir)

  ###############################################################################

  ensemble = Ensemble(metrix, newdata_minusEP, bbbb)

