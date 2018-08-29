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
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.tree import export_graphviz
from datetime import datetime
from sklearn.externals import joblib

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

###############################################################################
#
#  class for ML using random forest with randomised search
#
###############################################################################

class RandomForestGridSearch(object):
  '''This class is the doing the actual work in the following steps:
     * define smaller data frames: database, man_add, transform
     * split the data into training and test set
     * setup and run a grid search for best paramaters to define a decsion tree
     * create a new tree with best parameters
     * predict on this new tree with test data and cross-validated training data
     * analyse the predisctions with graphs and stats
  '''
  def __init__(self, metrix, outdir):
    self.metrix=metrix
    self.outdir=outdir
    self.prepare_metrix_data()
    self.split_data()
    self.grid_search()
    self.forest_best_params()
    self.predict()
    self.analysis()

  ###############################################################################
  #
  #  creating 3 data frames specific to the three development milestones I had
  #  1--> directly from data processing
  #  2--> after adding protein information
  #  3--> carrying out some further column transformations
  #
  ###############################################################################

  def prepare_metrix_data(self):
    '''Function to create smaller dataframes for directly after dataprocessing, after
       adding some protein information and after carrying out some custom solumn
       transformations.
    ******
    Input: large data frame
    Output: smaller dataframes; database, man_add, transform
    '''
    print('*' *80)
    print('*    Preparing input dataframes metrix_database, metrix_man_add, metrix_transform')
    print('*' *80)

    #look at the data that is coming from processing
    attr_database = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
                      'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
                      'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
                      'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF']
    metrix_database = self.metrix[attr_database]
    
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_database with following attributes %s \n' %(attr_database))

    #database plus manually added data
    attr_man_add = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
                    'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
                    'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
                    'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF',
                    'wavelength', 'Vcell', 'Matth_coeff', 'No_atom_chain', 'solvent_content',
                    'No_mol_ASU', 'MW_chain', 'sites_ASU']
    metrix_man_add = self.metrix[attr_man_add]

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_man_add with following attributes %s \n' %(attr_man_add))

    #after column transformation expected feature list
    attr_transform = ['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
                      'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
                      'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
                      'highreslimit', 'wilsonbfactor', 'anomalousslope',
                      'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
                      'diffF', 'wavelength', 'wavelength**3', 'wavelength**3/Vcell',
                      'Vcell', 'solvent_content', 'Vcell/Vm<Ma>', 'Matth_coeff',
                      'MW_ASU/sites_ASU/solvent_content', 'MW_chain', 'No_atom_chain',
                      'No_mol_ASU', 'MW_ASU', 'sites_ASU', 'MW_ASU/sites_ASU',
                      'MW_chain/No_atom_chain', 'wilson', 'bragg', 'volume_wilsonB_highres']                          

    metrix_transform = metrix_man_add.copy()

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_transform with following attributes %s \n' %(attr_transform))

    #column transformation
    #MW_ASU
    metrix_transform['MW_ASU'] = metrix_transform['MW_chain'] * metrix_transform['No_mol_ASU']

    #MW_ASU/sites_ASU
    metrix_transform['MW_ASU/sites_ASU'] = metrix_transform['MW_ASU'] / metrix_transform['sites_ASU']

    #MW_chain/No_atom_chain
    metrix_transform['MW_chain/No_atom_chain'] = metrix_transform['MW_chain'] / metrix_transform['No_atom_chain']

    #MW_ASU/sites_ASU/solvent_content
    metrix_transform['MW_ASU/sites_ASU/solvent_content'] = metrix_transform['MW_ASU/sites_ASU'] / metrix_transform['solvent_content']

    #wavelength**3
    metrix_transform['wavelength**3'] = metrix_transform['wavelength'] ** 3

    #wavelenght**3/Vcell
    metrix_transform['wavelength**3/Vcell'] = metrix_transform['wavelength**3'] / metrix_transform['Vcell']

    #Vcell/Vm<Ma>
    metrix_transform['Vcell/Vm<Ma>'] = metrix_transform['Vcell'] / (metrix_transform['Matth_coeff'] * metrix_transform['MW_chain/No_atom_chain'])

    #wilson
    metrix_transform['wilson'] = -2 * metrix_transform['wilsonbfactor']

    #bragg
    metrix_transform['bragg'] = (1 / metrix_transform['highreslimit'])**2

    #use np.exp to work with series object
    metrix_transform['volume_wilsonB_highres'] = metrix_transform['Vcell/Vm<Ma>'] * np.exp(metrix_transform['wilson'] * metrix_transform['bragg'])
    self.X_database = metrix_database
    self.X_man_add = metrix_man_add
    self.X_transform = metrix_transform

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_database, metrix_man_add, metrix_transform \n')

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

    y = self.metrix['EP_success']
    X_database_train, X_database_test, y_train, y_test = train_test_split(self.X_database, y, test_size=0.2, random_state=42)
    X_man_add_train, X_man_add_test, y_train, y_test = train_test_split(self.X_man_add, y, test_size=0.2, random_state=42)
    X_transform_train, X_transform_test, y_train, y_test = train_test_split(self.X_transform, y, test_size=0.2, random_state=42)
    assert self.X_database.columns.all() == X_database_train.columns.all()
    assert self.X_man_add.columns.all() == X_man_add_train.columns.all()
    assert self.X_transform.columns.all() == X_transform_train.columns.all()
    self.X_database_train = X_database_train
    self.X_man_add_train = X_man_add_train
    self.X_transform_train = X_transform_train
    self.X_database_test = X_database_test
    self.X_man_add_test = X_man_add_test
    self.X_transform_test = X_transform_test
    self.y_train = y_train
    self.y_test = y_test

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_database: X_database_train, X_database_test \n')
      text_file.write('metrix_man_add: X_man_add_train, X_man_add_test \n')
      text_file.write('metrix_transform: X_transform_train, X_transform_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')

    ###############################################################################
    #
    #  grid search for best parameter combination
    #
    ###############################################################################

  def grid_search(self):
  '''running a randomized search to find the parameter combination for a decision tree
     which gives the best accuracy score'''
    print('*' *80)
    print('*    Running GridSearch for best parameter combination for RandomForest')
    print('*' *80)

    #create the decision forest
    forest_clf_grid = RandomForestClassifier(random_state=42)

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Created random forest: forest_clf_grid \n')

    #set up grid search
    param_grid = {"criterion": ["gini", "entropy"],
                 'n_estimators': [10, 50, 100, 500],
                 'max_features': [1, 2, 4, 8, 16],
                 "min_samples_split": [5, 10, 15],
                 "max_depth": [3, 4, 5, 6],
                 "min_samples_leaf": [2, 4, 6],
                 "max_leaf_nodes": [5, 10, 15]}

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Running grid search for the following parameters: %s \n' %param_grid)
      text_file.write('use cv=10, scoring=accuracy \n')

    #building and running the grid search
    grid_search = GridSearchCV(forest_clf_grid, param_grid, cv=10,
                              scoring='accuracy')

    grid_search.fit(self.X_transform_train, self.y_train)

    #get best parameter combination and its score as accuracy
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Best parameters: ' +str(grid_search.best_params_)+'\n')
      text_file.write('Best score: ' +str(grid_search.best_score_)+'\n')
    
    feature_importances = grid_search.best_estimator_.feature_importances_
    feature_importances_ls = sorted(zip(feature_importances, self.X_transform_train), reverse=True)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Feature importances: %s \n' %feature_importances_ls)
    
    self.best_params = grid_search.best_params_

    ###############################################################################
    #
    #  creating new forest with best parameter combination
    #
    ###############################################################################

  def forest_best_params(self):
    '''create a new random forest using the best parameter combination found above'''
    print('*' *80)
    print('*    Building new forest based on best parameter combination')
    print('*' *80)

    self.forest_clf_grid_new = RandomForestClassifier(**self.best_params, random_state=42)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Created new decision forest "forest_clf_grid_new" using best parameters \n')

    print('*' *80)
    print('*    Saving new forest based on best parameter combination as pickle')
    print('*' *80)

    joblib.dump(self.forest_clf_grid_new, os.path.join(self.outdir,'best_forest_grid_search.pkl'))
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Creating pickle file for best forest as best_forest_grid_search.pkl \n')

    #visualise best decision tree
    trees = forest_clf_grid_new.estimators_
    i_tree = 0
    for tree in trees:
      with open(os.path.join(self.outdir,'forest_clf_grid_new_tree' + str(i_tree) + '.dot'), 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=X_transform_train.columns,
                   rounded=True, filled=True)
        f.close()
      dotfile = os.path.join(self.outdir, 'forest_clf_grid_new_tree' + str(i_tree) + '.dot')
      pngfile = os.path.join(self.outdir, 'forest_clf_grid_new_tree' + str(i_tree) + '.png')
      command = ["dot", "-Tpng", dotfile, "-o", pngfile]
      subprocess.check_call(command)
      i_tree = i_tree + 1

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Writing DOTfile and convert to PNG for "forest_clf_grid_new" \n')
      text_file.write('DOT filename: forest_clf_grid_new.dot \n')
      text_file.write('PNG filename: forest_clf_grid_new.png \n')

    print('*' *80)
    print('*    Getting basic stats for new forest')
    print('*' *80)

    #not the best measure to use as it heavily depends on the sample 
    #distribution --> accuracy
    accuracy_each_cv = cross_val_score(self.forest_clf_grid_new, self.X_transform_train, self.y_train,
                    cv=10, scoring='accuracy')
    accuracy_mean_cv = cross_val_score(self.forest_clf_grid_new, self.X_transform_train, self.y_train,
                    cv=10, scoring='accuracy').mean()
    # calculate cross_val_scoring with different scoring functions for CV train set
    train_roc_auc = cross_val_score(self.forest_clf_grid_new, self.X_transform_train, self.y_train, cv=10,
                    scoring='roc_auc').mean()
    train_accuracy = cross_val_score(self.forest_clf_grid_new, self.X_transform_train, self.y_train, cv=10,
                    scoring='accuracy').mean()
    train_recall = cross_val_score(self.forest_clf_grid_new, self.X_transform_train, self.y_train, cv=10,
                    scoring='recall').mean()
    train_precision = cross_val_score(self.forest_clf_grid_new, self.X_transform_train, self.y_train, cv=10,
                    scoring='precision').mean()
    train_f1 = cross_val_score(self.forest_clf_grid_new, self.X_transform_train, self.y_train, cv=10,

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Accuracy for each of 10 CV folds: %s \n' %accuracy_each_cv)
      text_file.write('Mean accuracy over all 10 CV folds: %s \n' %accuracy_mean_cv)
      text_file.write('ROC_AUC mean for 10-fold CV: %s \n' %train_roc_auc)
      text_file.write('Accuracy mean for 10-fold CV: %s \n' %train_accuracy)
      text_file.write('Recall mean for 10-fold CV: %s \n' %train_recall)
      text_file.write('Precision mean for 10-fold CV: %s \n' %train_precision)
      text_file.write('F1 score mean for 10-fold CV: %s \n' %train_f1)

    ###############################################################################
    #
    #  Predicting with test set and cross-validation set using the bets forest
    #
    ###############################################################################

  def predict(self):
    '''do predictions using the best tree an the test set as well as training set with
       10 cross-validation folds and doing some initial analysis on the output'''
    print('*' *80)
    print('*    Predict using new forest and test/train_CV set')
    print('*' *80)

    #try out how well the classifier works to predict from the test set
    self.y_pred_class = self.forest_clf_grid_new.predict(self.X_transform_test)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_transform_test in y_pred_class \n')

    #alternative way to not have to use the test set
    self.y_train_pred = cross_val_predict(self.forest_clf_grid_new, self.X_transform_train, self.y_train,
                      cv=10)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_transform_train with 10-fold CV in y_train_pred \n')

    print('*' *80)
    print('*    Calculate prediction stats')
    print('*' *80)

    # calculate accuracy
    y_accuracy = metrics.accuracy_score(self.y_test, self.y_pred_class)

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

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Accuracy score or agreement between y_test and y_pred_class: %s \n' %y_accuracy)
      text_file.write('Class distribution for y_test: %s \n' %class_dist)
      text_file.write('Percent 1s in y_test: %s \n' %ones)
      text_file.write('Percent 0s in y_test: %s \n' %zeros)
      text_file.write('Null accuracy in y_test: %s \n' %null_acc)

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

    datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    # IMPORTANT: first argument is true values, second argument is predicted values
    # this produces a 2x2 numpy array (matrix)
    conf_mat_test = metrics.confusion_matrix(self.y_test, self.y_pred_class)
    conf_mat_10CV = metrics.confusion_matrix(self.y_train, self.y_train_pred)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('confusion matrix using test set: %s \n' %conf_mat_test)
      text_file.write('confusion matrix using 10-fold CV: %s \n' %conf_mat_10CV)

    # slice confusion matrix into four pieces
    #[row, column] for test set
    TP = conf_mat_test[1, 1]
    TN = conf_mat_test[0, 0]
    FP = conf_mat_test[0, 1]
    FN = conf_mat_test[1, 0]
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Slicing confusion matrix for test set into: TP, TN, FP, FN \n')

    #[row, column] for CV train set
    TP_CV = conf_mat_10CV[1, 1]
    TN_CV = conf_mat_10CV[0, 0]
    FP_CV = conf_mat_10CV[0, 1]
    FN_CV = conf_mat_10CV[1, 0]
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Slicing confusion matrix for 10-fold CV into: TP_CV, TN_CV, FP_CV, FN_CV \n')

    #metrics calculated from confusion matrix
    # use float to perform true division, not integer division
    acc_score_man_test = (TP + TN) / float(TP + TN + FP + FN)
    acc_score_sklearn_test = metrics.accuracy_score(self.y_test, self.y_pred_class)
    acc_score_man_CV = (TP_CV + TN_CV) / float(TP_CV + TN_CV + FP_CV + FN_CV)
    acc_score_sklearn_CV = metrics.accuracy_score(self.y_train, self.y_train_pred)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Accuracy score: \n')
      text_file.write('accuracy score manual test: %s \n' %acc_score_man_test)
      text_file.write('accuracy score sklearn test: %s \n' %acc_score_sklearn_test)
      text_file.write('accuracy score manual CV: %s \n' %acc_score_man_CV)
      text_file.write('accuracy score sklearn CV: %s \n' %acc_score_sklearn_CV)

    #something of one class put into the other
    class_err_man_test = (FP + FN) / float(TP + TN + FP + FN)
    class_err_sklearn_test = 1 - metrics.accuracy_score(self.y_test, self.y_pred_class)
    class_err_man_CV = (FP_CV + FN_CV) / float(TP_CV + TN_CV + FP_CV + FN_CV)
    class_err_sklearn_CV = 1 - metrics.accuracy_score(self.y_train, self.y_train_pred)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Classification error: \n')  
      text_file.write('classification error manual test: %s \n' %class_err_man_test)
      text_file.write('classification error sklearn test: %s \n' %class_err_sklearn_test)
      text_file.write('classification error manual CV: %s \n' %class_err_man_CV)
      text_file.write('classification error sklearn CV: %s \n' %class_err_sklearn_CV)

    #same as recall or true positive rate; correctly placed positive cases
    sensitivity_man_test = TP / float(FN + TP)
    sensitivity_sklearn_test = metrics.recall_score(self.y_test, self.y_pred_class)
    sensitivity_man_CV = TP_CV / float(FN_CV + TP_CV)
    sensitivity_sklearn_CV = metrics.recall_score(self.y_train, self.y_train_pred)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Sensitivity/Recall/True positives: \n')
      text_file.write('sensitivity manual test: %s \n' %sensitivity_man_test)
      text_file.write('sensitivity sklearn test: %s \n' %sensitivity_sklearn_test)
      text_file.write('sensitivity manual CV: %s \n' %sensitivity_man_CV)
      text_file.write('sensitivity sklearn CV: %s \n' %sensitivity_sklearn_CV)
  
    #calculate specificity
    specificity_man_test = TN / (TN + FP)
    specificity_man_CV = TN_CV / (TN_CV + FP_CV)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Specificity: \n')
      text_file.write('specificity manual test: %s \n' %specificity_man_test)
      text_file.write('specificity manual CV: %s \n' %specificity_man_CV)
    
    #calculate false positive rate
    false_positive_rate_man_test = FP / float(TN + FP)
    false_positive_rate_man_CV = FP_CV / float(TN_CV + FP_CV)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('False positive rate or 1-specificity: \n')
      text_file.write('false positive rate manual test: %s \n' %false_positive_rate_man_test)
      text_file.write('1 - specificity test: %s \n' %(1 - specificity_man_test))
      text_file.write('false positive rate manual CV: %s \n' %false_positive_rate_man_CV)
      text_file.write('1 - specificity CV: %s \n' %(1 - specificity_man_CV))

    #calculate precision or how confidently the correct placement was done
    precision_man_test = TP / float(TP + FP)
    precision_sklearn_test = metrics.precision_score(self.y_test, self.y_pred_class)
    precision_man_CV = TP_CV / float(TP_CV + FP_CV)
    precision_sklearn_CV = metrics.precision_score(self.y_train, self.y_train_pred)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Precision or confidence of classification: \n')
      text_file.write('precision manual: %s \n' %precision_man_test)
      text_file.write('precision sklearn: %s \n' %precision_sklearn_test)
      text_file.write('precision manual CV: %s \n' %precision_man_CV)
      text_file.write('precision sklearn CV: %s \n' %precision_sklearn_CV)

    #F1 score; uses precision and recall
    f1_score_sklearn_test = f1_score(self.y_test, self.y_pred_class)
    f1_score_sklearn_CV = f1_score(self.y_train, self.y_train_pred)
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('F1 score: \n')
      text_file.write('F1 score sklearn test: %s \n' %f1_score_sklearn_test)
      text_file.write('F1 score sklearn CV: %s \n' %f1_score_sklearn_CV)

    #probabilities of predicting y_train with X_transform_train using 10-fold CV
    self.y_pred_proba_train_CV = cross_val_predict(self.forest_clf_grid_new, self.X_transform_train, self.y_train, cv=10, method='predict_proba')

    #probabilities of predicting y_test with X_transform_test
    self.y_pred_proba_test = self.forest_clf_grid_new.predict_proba(self.X_transform_test)
    
#    self.y_scores=self.forest_clf_grid_grid_new.predict_proba(self.X_transform_train)#train set
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Storing prediction probabilities for X_transform_train and y_train with 10-fold CV in y_pred_proba_train_CV \n')
      text_file.write('Storing prediction probabilities for X_transform_test and y_test in y_pred_proba_test \n')

    # 8 bins for prediction probability on the test set
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Plotting histogram for y_pred_proba_train_CV \n')
      text_file.write('Plotting histogram for y_pred_proba_test \n')
      
    #plot histograms of probabilities  
    def plot_hist_pred_proba(y_pred_proba, name):
      plt.hist(y_pred_proba, bins=8)
      plt.xlim(0,1)
      plt.title('Histogram of predicted probabilities for y_pred_proba_%s to be class 1' %name)
      plt.xlabel('Predicted probability of EP_success')
      plt.ylabel('Frequency')
      plt.savefig(os.path.join(self.outdir, 'hist_pred_proba_forest_grid_'+name+datestring+'.png'))
      plt.close()

    plot_hist_pred_proba(self.y_pred_proba_train_CV[:, 1], 'train_CV_')
    plot_hist_pred_proba(self.y_pred_proba_test[:, 1], 'test_')

    #get y_scores for the predictions to be bale to plot ROC curve
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Getting y_scores for y_pred_proba_train_CV and y_pred_proba_test as y_scores_train_CV and y_scores_test\n')

    # store the predicted probabilities for class 1
    self.y_scores_train_CV = self.y_pred_proba_train_CV[:, 1]
    self.y_scores_test = self.y_pred_proba_test[:, 1]

    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Plotting Precision-Recall for y_test and y_scores_test \n')
      text_file.write('Plotting Precision-Recall for y_train and y_scores_train_CV \n')

    #plot precision and recall curve
    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest, name):
        plt.plot(thresholds_forest, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds_forest, recalls[:-1], "g--", label="Recall")
        plt.title('Precsion-Recall plot for for EP_success classifier using %s set' %name)
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0,1])
        plt.savefig(os.path.join(self.outdir, 'Precision_Recall_forest_grid_'+name+datestring+'.png'))
        plt.close()
        
    #plot Precision Recall Threshold curve for test set 
    precisions, recalls, thresholds_forest = precision_recall_curve(self.y_test, self.y_scores_test)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest, 'test_')

    #plot Precision Recall Threshold curve for CV train set 
    precisions, recalls, thresholds_forest = precision_recall_curve(self.y_train, self.y_scores_train_CV)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest, 'train_CV_')

    # IMPORTANT: first argument is true values, second argument is predicted probabilities

    # we pass y_test and y_pred_prob
    # we do not use y_pred_class, because it will give incorrect results without generating an error
    # roc_curve returns 3 objects fpr, tpr, thresholds
    # fpr: false positive rate
    # tpr: true positive rate
    
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('Plotting ROC curve for y_test and y_scores_test \n')
      text_file.write('Plotting ROC curve for y_train and y_scores_train_CV \n')
    
    #plot ROC curves
    def plot_roc_curve(fpr, tpr, name, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.title('ROC curve for EP_success classifier using %s set' %name) 
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        plt.savefig(os.path.join(self.outdir, 'ROC_curve_forest_grid_'+name+datestring+'.png'))
        plt.close()
        
    #ROC curve for test set      
    fpr, tpr, thresholds = roc_curve(self.y_test, self.y_scores_test)#test set
    plot_roc_curve(fpr, tpr, 'test_')

    #ROC curve for 10-fold CV train set
    fpr_CV, tpr_CV, thresholds_CV = roc_curve(self.y_train, self.y_scores_train_CV)#CV train set
    plot_roc_curve(fpr_CV, tpr_CV, 'train_CV_')

    #calculate the area under the curve to get the performance for a classifier
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    AUC_test = metrics.roc_auc_score(self.y_test, self.y_pred_proba_test[:, 1])
    AUC_train_CV = metrics.roc_auc_score(self.y_train, self.y_pred_proba_train_CV[:, 1])
    with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
      text_file.write('AUC for test set: %s \n' %AUC_test)
      text_file.write('AUC for CV train set: %s \n' %AUC_train_CV)

    # define a function that accepts a threshold and prints sensitivity and specificity
    def evaluate_threshold(tpr, fpr, thresholds, threshold, name):
      sensitivity = tpr[thresholds > threshold][-1]
      specificity = 1 - fpr[thresholds > threshold][-1]
      with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
        text_file.write('Sensitivity for %s at threshold %.2f: %s \n' %(name, threshold, sensitivity))
        text_file.write('Specificity for %s at threshold %.2f: %s \n' %(name, threshold, specificity))
    
    evaluate_threshold(tpr, fpr, thresholds, 0.5, 'test_')
    evaluate_threshold(tpr, fpr, thresholds, 0.4, 'test_')
    evaluate_threshold(tpr_CV, fpr_CV, thresholds_CV, 0.5, 'train_CV')
    evaluate_threshold(tpr_CV, fpr_CV, thresholds_CV, 0.4, 'train_CV')
    
    def scoring(X, y, name, cv ):
      # calculate cross_val_scores with different scoring functions for test set
      roc_auc = cross_val_score(self.forest_clf_grid_new, X, y, cv=cv,
                      scoring='roc_auc').mean()
      accuracy = cross_val_score(self.forest_clf_grid_new, X, y, cv=cv,
                      scoring='accuracy').mean()
      recall = cross_val_score(self.forest_clf_grid_new, X, y, cv=cv,
                      scoring='recall').mean()
      precision = cross_val_score(self.forest_clf_grid_new, X, y, cv=cv,
                      scoring='precision').mean()
      f1 = cross_val_score(self.forest_clf_grid_new, X, y, cv=cv,
                      scoring='f1').mean()
      with open(os.path.join(self.outdir, 'randomforest_gridsearch.txt'), 'a') as text_file:
        text_file.write('ROC_AUC for %s: %s \n' %(name, roc_auc))
        text_file.write('Accuracy for %s: %s \n' %(name, accuracy))
        text_file.write('Recall for %s: %s \n' %(name, recall))
        text_file.write('Precision for %s: %s \n' %(name, precision))
        text_file.write('F1 score for %s: %s \n' %(name, f1))

    scoring(self.X_transform_test, self.y_test, 'test', cv=None)
    scoring(self.X_transform_train, self.y_train, 'train_CV', cv=10)

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  ###############################################################################

  random_forest_grid_search = RandomForestGridSearch(metrix, args.outdir)

