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
import imblearn
import joblib
import logging

import matplotlib
matplotlib.use("Agg")

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from datetime import datetime
from scipy.stats import randint
from scipy.stats import uniform
from math import pi
from pathlib import Path

###############################################################################
#
#  set up logging
#
###############################################################################

logging.basicConfig(level=logging.INFO, filename = "training.log", filemode = "w")

###############################################################################
#
#  define command line arguments
#
###############################################################################

def parse_command_line():
  '''defining the command line input to make it runable'''
  parser = argparse.ArgumentParser(
                      description = "AdaBoost and DecisionTree published hyperparameters")

  parser.add_argument(
    "--input", 
    type = str, 
    dest = "input",
    default = "",
    help = "The input CSV file")

  parser.add_argument(
    "--outdir",
    type = str,
    dest = "outdir",
    default = "",
    help = "Specify output directory")

  args = parser.parse_args()
  if args.input == "":
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
  # Load training files
  training_dir_path = Path(csv_path)
  assert (
  training_dir_path.exists()
  ), f"Could not find directory at {training_dir_path}"
  logging.info(f"Opened dataframe containing training data")
  return pd.read_csv(csv_path)
  
def make_output_folder(outdir):
  output_dir = os.path.join(outdir, "decisiontree_ada_published")
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

###############################################################################
#
#  class for ML using random forest with randomised search and Ada boosting
#
###############################################################################

class RandomForestAdaRandSearch(object):
  '''This class is the doing the actual work in the following steps:
     * define smaller data frames: database, man_add, transform
     * split the data into training and test set
     * setup and run a randomized search for best paramaters to define a random forest
     * create a new random forest with best parameters
     * predict on this new random forest with test data and cross-validated training data
     * analyse the predisctions with graphs and stats
  '''
  def __init__(self, metrix, output_dir):
    self.metrix = metrix
    self.output_dir = output_dir
    self.prepare_metrix_data()
    self.split_data()
    self.forest_best_params()
    self.predict()
    self.analysis()

  def prepare_metrix_data(self):
    '''Function to create smaller dataframe.
    ******
    Input: large data frame
    Output: smaller dataframe
    '''
    print("*" * 80)
    print("*    Preparing input dataframe")
    print("*" * 80)

    autobuild_columns = ["anomalousCC",
                         "anomalousslope",
                         "lowreslimit",
                         "f",
                         "diffF",
                         "diffI",
                         "autobuild_success"]

    self.data = self.metrix[autobuild_columns]

    logging.info(f"Using dataframe with column labels {autobuild_columns}")

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
    print("*" * 80)
    print("*    Splitting data into test and training set with test=20%")
    print("*" * 80)

    y = self.metrix["autobuild_success"]
    X = self.data[["anomalousCC",
                   "anomalousslope",
                   "lowreslimit",
                   "f",
                   "diffF",
                   "diffI"]]

#stratified split of samples
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)

    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test

    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    y_train_shape = y_train.shape
    y_test_shape = y_test.shape

    logging.info(f"Shape of test data X_train {X_train_shape}")
    logging.info(f"Shape of test data X_test {X_test_shape}")
    logging.info(f"Shape of test data y_train {y_train_shape}")
    logging.info(f"Shape of test data y_test {y_test_shape}")


###############################################################################
#
#  optional step of over/undersampling if there is a large mis-match between classes
#
###############################################################################

   #the weight distribution for the classes used by "class_weight" weights = {0:0.1, 1:0.9}

    #print('*' *80)
    #print('*    Applying Over/Undersampling and SMOTE')
    #print('*' *80)

    #oversample = RandomOverSampler(sampling_strategy = 'minority')
    #oversample = RandomOverSampler(sampling_strategy = 0.1)
    #oversample = SMOTE(sampling_strategy = 0.3, random_state=28)
    # fit and apply the transform
    #X_over, y_over = oversample.fit_resample(self.X_newdata_transform_train, self.y_train)

    #undersample = RandomUnderSampler(sampling_strategy=0.7)
    #X_over, y_over = undersample.fit_resample(X_over, y_over)
    #self.X_over = X_over
    #self.y_over = y_over

###############################################################################
#
#  creating classifier with best parameter from IUCrJ publication
#
###############################################################################

  def forest_best_params(self):
    '''create a new random forest using the best parameter combination found above'''
    print("*" * 80)
    print("*    Building new forest based on best parameter combination and save as pickle")
    print("*" * 80)

# a blank decision tree with Ada Boost that can be used for hyperparameter search when
# when starting from scratch
#    clf2 = DecisionTreeClassifier(**self.best_params_base_estimator,
#                                  random_state= 0)
#    self.tree_clf2_new_rand = AdaBoostClassifier(clf2,
#                                                 **self.best_params_ada, 
#                                                 algorithm ="SAMME.R",
#                                                 random_state=100)

# hyperparameters as were used for the classifier published in IUCrJ; this was first run
# in deployment with really bad performance;
# the saved model is named: 2019 calibrated_classifier_20190501_1115.pkl
    clf2 = DecisionTreeClassifier(criterion="entropy",
                                  max_depth=3,
                                  max_features=2,
                                  max_leaf_nodes=17,
                                  min_samples_leaf=8,
                                  min_samples_split=18,
                                  random_state= 0,
                                  class_weight = "balanced")
    self.tree_clf2_new_rand = AdaBoostClassifier(clf2,
                                                 learning_rate=0.6355,
                                                 n_estimators=5694,
                                                 algorithm ="SAMME.R",
                                                 random_state=5)

# hyperparameters for a new classifier; this one was found after adding some user data
# from run1 2020 to the training data; this one is now running in the automated data
# analysis pipelines; the saved model is named: calibrated_classifier_20200408_1552.pkl
#    clf2 = DecisionTreeClassifier(criterion="entropy",
#                                  max_depth=5,
#                                  max_features=2,
#                                  max_leaf_nodes=15,
#                                  min_samples_leaf=5,
#                                  min_samples_split=3,
#                                  random_state= 0,
#                                  class_weight = "balanced")
#    self.tree_clf2_new_rand = AdaBoostClassifier(
#                                           clf2,
#                                           learning_rate=0.6846,
#                                           n_estimators=4693,
#                                           algorithm ="SAMME.R",
#                                           random_state=5)

    classifier_params = self.tree_clf2_new_rand.get_params()
    print(classifier_params)

    self.tree_clf2_new_rand.fit(self.X_train, self.y_train)

    logging.info(
           f"Created classifier based on IUCrJ publication and fitted training data.\n"
           f"Classifier parameters: {classifier_params}")

###############################################################################
#
#  Bootstrapping to find the 95% confidence interval
#
###############################################################################

    # Trying some bootstrap to assess confidence interval for classification
    print("*" * 80)
    print("*    Calculating confidence interval for best decisiontree with AdaBoost")
    print("*" * 80)

    def bootstrap_calc(data_train, data_test, train_labels, test_labels, found_model):
      # configure bootstrap
      n_iterations =  1000
      n_size = int(len(data_train))

      # run bootstrap
      stats = list()
      for i in range(n_iterations):
        # prepare train and test sets
        train_boot = resample(data_train, n_samples = n_size)
        test_boot = train_labels
        # fit model
        model = found_model
        model.fit(train_boot, test_boot)
        # evaluate model
        predictions = model.predict(data_test)
        score = accuracy_score(test_labels, predictions)
        stats.append(score)

      # plot scores
      plt.hist(stats)
      plt.savefig(os.path.join(self.output_dir, "bootstrap_hist_ada.png"), dpi=600)
      plt.close()
      # confidence interval
      alpha = 0.95
      p = ((1.0 - alpha) / 2.0) * 100
      lower = max(0.0, np.percentile(stats, p))
      p = (alpha + ((1.0 - alpha) / 2.0)) * 100
      upper = min(1.0, np.percentile(stats, p))
      
      lower_boundary = round((lower * 100), 2)
      upper_boundary = round((upper * 100), 2)
      
      logging.info(f"Calculating 95% confidence interval from bootstrap exercise\n"
                   f"Lower boundary: {lower_boundary}\n"
                   f"Upper boundary: {upper_boundary}")

    bootstrap_calc(self.X_train,
                   self.X_test,
                   self.y_train,
                   self.y_test,
                   self.tree_clf2_new_rand)

###############################################################################
#
#  get feature importances for best tree and full classifier;
#  plot feature importances for both
#
###############################################################################

    #print(self.tree_clf2_new_rand.estimators_)
    #print(self.tree_clf2_new_rand.feature_importances_)
    
    attr = ["anomalousCC",
            "anomalousslope",
            "lowreslimit",
            "f",
            "diffF",
            "diffI"]
    
    feature_importances = self.tree_clf2_new_rand.feature_importances_
    feature_importances_ls = sorted(zip(feature_importances, attr),
                                              reverse = True)
    #print(feature_importances_transform_ls)
    feature_importances_tree_mean = np.mean(
             [tree.feature_importances_ for tree in self.tree_clf2_new_rand.estimators_],
             axis = 0)

    feature_importances_tree_mean_ls = sorted(zip(feature_importances_tree_mean, attr),
                                              reverse = True)
    logging.info(
          f"Feature importances, for best tree in classifier: {feature_importances_ls}\n"
          f"Plotting bar plot of feature importances for best tree in classifier\n"
          f"Feature importances, mean over all trees: {feature_importances_tree_mean_ls}\n"
          f"Plotting bar plot of feature importances with mean and std for classifier")

    def feature_importances_best_estimator(feature_list, directory):
      datestring = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
      feature_list.sort(key = lambda x: x[1], reverse = True)
      feature = list(zip(*feature_list))[1]
      score = list(zip(*feature_list))[0]
      x_pos = np.arange(len(feature))
      plt.bar(x_pos, score,align="center")
      plt.xticks(x_pos, feature, rotation = 90, fontsize = 18)
      plt.title("Histogram of Feature Importances for best tree in best classifier")
      plt.xlabel("Features")
      plt.tight_layout()
      plt.savefig(os.path.join(directory,
              "feature_importances_besttree_bestclassifier_bar_plot_"+datestring+".png"),
              dpi = 600)
      plt.close()
    
    feature_importances_best_estimator(feature_importances_ls,
                                       self.output_dir)

    def feature_importances_pandas(clf, X_train, directory):   
      datestring = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")      
      feature_list = []
      for tree in clf.estimators_:
        feature_importances_ls = tree.feature_importances_
        feature_list.append(feature_importances_ls)

      df = pd.DataFrame(feature_list, columns = X_train.columns)
      df_mean = df[X_train.columns].mean(axis = 0)
      df_std = df[X_train.columns].std(axis = 0)
      df_mean.plot(kind = "bar", color = "b", yerr = [df_std],
                   align = "center", figsize = (20,10), rot = 90, fontsize = 18)
      plt.title(
            "Histogram of Feature Importances over all trees in best classifier with std")
      plt.xlabel('Features')
      plt.tight_layout()
      plt.savefig(os.path.join(directory,
      "feature_importances_mean_std_bestclassifier_bar_plot_"+datestring+".png"), dpi = 600)
      plt.close()
      
    feature_importances_pandas(self.tree_clf2_new_rand,
                               self.X_train,
                               self.output_dir)
    #feature_importances_pandas(self.tree_clf_rand_ada_new_transform, self.X_over, 'newdata_minusEP', self.newdata_minusEP)

###############################################################################
#
#  save best classifier as pickle file for future use
#
###############################################################################

    def write_pickle(forest, directory):
      datestring = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
      joblib.dump(forest,
                   os.path.join(directory, "best_classifier_rand_ada_"+datestring+".pkl"))
    
    write_pickle(self.tree_clf2_new_rand,
                 self.output_dir)

    logging.info(f"Saving best classifier.")

    print("*" * 80)
    print("*    Getting basic stats for new forest")
    print("*" * 80)

###############################################################################
#
#  get basic stats for 3-fold cross-validation on the training data
#
###############################################################################

    def basic_stats(forest, data_train, labels_train, directory):
      #distribution --> accuracy
      accuracy_each_cv = cross_val_score(forest, data_train,
                                         labels_train, cv=3, scoring="accuracy")
      accuracy_mean_cv = round(cross_val_score(forest, data_train,
                        labels_train, cv=3, scoring="accuracy").mean(), 4)
      ## calculate cross_val_scoring with different scoring functions for CV train set
      train_roc_auc = round(cross_val_score(forest, data_train,
                                      labels_train, cv=3, scoring="roc_auc").mean(), 4)
      train_recall = round(cross_val_score(forest, data_train,
                                     labels_train, cv=3, scoring="recall").mean(), 4)
      train_precision = round(cross_val_score(forest, data_train,
                                        labels_train, cv=3, scoring="precision").mean(), 4)
      train_f1 = round(cross_val_score(forest, data_train,
                                 labels_train, cv=3, scoring="f1").mean(), 4)

      logging.info(
        f"Get various cross_val_scores to evaluate clf performance for best parameters\n"
        f"Training accuracy for individual folds in 3-fold CV: {accuracy_each_cv}\n"
        f"Mean training accuracy over all folds in 3-fold CV: {accuracy_mean_cv}\n"
        f"Mean training recall for 3-fold CV: {train_recall}\n"
        f"Mean training precision for 3-fold CV: {train_precision}\n"
        f"Mean training ROC_AUC for 3-fold CV: {train_roc_auc}\n"
        f"Mean training F1 score for 3-fold CV: {train_f1}")

    basic_stats(self.tree_clf2_new_rand,
                self.X_train,
                self.y_train,
                self.output_dir)

###############################################################################
#
#  predicting with test set
#
###############################################################################

  def predict(self):
    '''do predictions using the best classifier and the test set and doing some
       initial analysis on the output'''
       
    print("*" * 80)
    print("*    Predict using new forest and test set")
    print("*" * 80)

    #try out how well the classifier works to predict from the test set
    self.y_pred = self.tree_clf2_new_rand.predict(self.X_test)
    self.y_pred_proba = self.tree_clf2_new_rand.predict_proba(self.X_test)
    self.y_pred_proba_ones = self.y_pred_proba[:, 1]#test data to be class 1
    self.y_pred_proba_zeros = self.y_pred_proba[:, 0]#test data to be class 0
    
    y_pred_csv = os.path.join(self.output_dir, "y_pred.csv")
    y_pred_proba_csv = os.path.join(self.output_dir, "y_pred_proba.csv")
    
    np.savetext(y_pred_csv, self.y_pred, delimiter = ",")
    np.savetext(y_pred_proba_csv, self.y_pred_proba, delimiter = ",")

#    with open(y_pred_csv, "w", newline="") as pred_csv:
#      pred_out = csv.writer(pred_csv)
#      pred_out.writerows(self.y_pred)

    logging.info(f"Storing predictions for test set to y_pred.\n"
                 f"Storing probabilities for predictions for the test set to y_pred_proba")

    print("*" * 80)
    print("*    Calculate prediction stats")
    print("*" * 80)

    def prediction_stats(y_test, y_pred, directory):
      # calculate accuracy
      y_accuracy = accuracy_score(y_test, y_pred)

      # examine the class distribution of the testing set (using a Pandas Series method)
      class_dist = self.y_test.value_counts()
      class_zero = class_dist[0]
      class_one = class_dist[1]

      self.biggest_class = 0
      if class_zero > class_one:
        self.biggest_class = class_zero
      else:
        self.biggest_class = class_one

      # calculate the percentage of ones
      # because y_test only contains ones and zeros,
      # we can simply calculate the mean = percentage of ones
      ones = round(y_test.mean(), 4)

      # calculate the percentage of zeros
      zeros = round(1 - y_test.mean(), 4)

      # calculate null accuracy in a single line of code
      # only for binary classification problems coded as 0/1
      null_acc = round(max(y_test.mean(), 1 - y_test.mean()), 4)

      logging.info(
          f"Accuracy score or agreement between y_test and y_pred: {y_accuracy}\n"
          f"Class distribution for y_test: {class_dist}\n"
          f"Percent 1s in y_test: {ones}\n"
          f"Percent 0s in y_test: {zeros}\n"
          f"Null accuracy in y_test: {null_acc}")

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

    print("*" * 80)
    print("*    Detailed analysis and plotting")
    print("*" * 80)

###############################################################################
#
#  calculate and draw confusion matrix for test set predictions
#
###############################################################################

    # IMPORTANT: first argument is true values, second argument is predicted values
    # this produces a 2x2 numpy array (matrix)

    conf_mat_test = confusion_matrix(self.y_test, self.y_pred)
    
    logging.info(f"confusion matrix using test set: {conf_mat_test}")
    def draw_conf_mat(matrix, directory):
      datestring = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
      labels = ["0", "1"]      
      ax = plt.subplot()
      sns.heatmap(matrix, annot = True, ax = ax,
                  annot_kws = {"size": 18}, vmin = 0, vmax = self.biggest_class)
      plt.title("Confusion matrix of the classifier")
      ax.set_xticklabels(labels, fontdict = {"fontsize": 18})
      ax.set_yticklabels(labels, fontdict = {"fontsize": 18})
      plt.xlabel("Predicted", fontsize = 20)
      plt.ylabel("True", fontsize = 20)
      plt.tight_layout()
      plt.savefig(os.path.join(directory,
                  "confusion_matrix_for_test_set_predictions"+datestring+".png"), dpi = 600)
      plt.close()

    draw_conf_mat(conf_mat_test,
                  self.output_dir)

###############################################################################
#
#  calculate stats for the test set using classification outcomes
#
###############################################################################

    TP = conf_mat_test[1, 1]
    TN = conf_mat_test[0, 0]
    FP = conf_mat_test[0, 1]
    FN = conf_mat_test[1, 0]
    
    logging.info(f"False-positives in predicting the test set: {FP}")
    logging.info(f"False-negatives in predicting the test set: {FN}")

    #calculate accuracy
    acc_score_man_test = round((TP + TN) / float(TP + TN + FP + FN), 4)
    acc_score_sklearn_test = round(accuracy_score(self.y_test, self.y_pred), 4)
    #classification error
    class_err_man_test = round((FP + FN) / float(TP + TN + FP + FN), 4)
    class_err_sklearn_test = round(1 - accuracy_score(self.y_test, self.y_pred), 4)
    #sensitivity/recall/true positive rate; correctly placed positive cases
    sensitivity_man_test = round(TP / float(FN + TP), 4)
    sensitivity_sklearn_test = round(recall_score(self.y_test, self.y_pred), 4)
    #specificity
    specificity_man_test = round(TN / (TN + FP), 4)
    #false positive rate
    false_positive_rate_man_test = round(FP / float(TN + FP), 4)
    #precision/confidence of placement
    precision_man_test = round(TP / float(TP + FP), 4)
    precision_sklearn_test = round(precision_score(self.y_test, self.y_pred), 4)
    #F1 score; uses precision and recall
    f1_score_sklearn_test = round(f1_score(self.y_test, self.y_pred), 4)

    logging.info(f"Detailed stats for the test set\n"
                 f"Accuracy score:\n"
                 f"accuracy score manual test: {acc_score_man_test}\n"
                 f"accuracy score sklearn test: {acc_score_sklearn_test}\n"
                 f"Classification error:\n"
                 f"classification error manual test: {class_err_man_test}\n"
                 f"classification error sklearn test: {class_err_sklearn_test}\n"
                 f"Sensitivity/Recall/True positives:\n"
                 f"sensitivity manual test: {sensitivity_man_test}\n"
                 f"sensitivity sklearn test: {sensitivity_sklearn_test}\n"
                 f"Specificity:\n"
                 f"specificity manual test: {specificity_man_test}\n"
                 f"False positive rate or 1-specificity:\n"
                 f"false positive rate manual test: {false_positive_rate_man_test}\n"
                 f"Precision or confidence of classification:\n"
                 f"precision manual: {precision_man_test}\n"
                 f"precision sklearn: {precision_sklearn_test}\n"
                 f"F1 score:\n"
                 f"F1 score sklearn test: {f1_score_sklearn_test}")
    
    data_dict = {"group" : "prediction",
                 "ACC (%)" : (acc_score_man_test * 100),
                 "Class Error (%)" : (class_err_man_test * 100),
                 "Sensitivity (%)" : (sensitivity_man_test * 100),
                 "Specificity (%)" : (specificity_man_test * 100),
                 "FPR (%)" : (false_positive_rate_man_test * 100),
                 "Precision (%)" : (precision_man_test * 100),
                 "F1 score (%)" : (f1_score_sklearn_test * 100)}
    
    df = pd.DataFrame(data = data_dict, index = [0])
    
    def plot_radar_chart(df, directory):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')

      # ------- PART 1: Create background

      # number of variable
      categories = list(df)[1:]
      print(categories)
      N = len(categories)

      # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
      angles = [n / float(N) * 2 * pi for n in range(N)]
      angles += angles[:1]

      # Initialise the spider plot
      #fig = plt.figure(figsize=(9, 9))
      fig = plt.figure(figsize=(7, 6))
      ax = fig.add_subplot(111, polar = True)

      # If you want the first axis to be on top:
      ax.set_theta_offset(pi / 2)
      ax.set_theta_direction(-1)

      # Draw one axe per variable + add labels labels yet
      ax.set_xticks(angles[:-1])
      ax.set_xticklabels(categories, fontsize = 20, wrap = True)
      #plt.xticks(angles[:-1], categories)

      # Draw ylabels
      ax.set_rlabel_position(15)
      ax.set_yticks([20, 40, 60, 80, 100])
      ax.set_yticklabels(["20", "40", "60", "80", "100%"], fontsize = 20, wrap = True)
      ax.set_ylim(0, 100)

      # ------- PART 2: Add plots
      #values = df.loc[0].values.flatten().tolist()
      values = df.loc[0].drop('group').values.flatten().tolist()
      print(values)
      values += values[:1]
      ax.plot(angles, values, linewidth = 2, linestyle = "solid", label = "Test set")
      ax.fill(angles, values, "b", alpha = 0.1)
      plt.savefig(os.path.join(directory,
                  "radar_chart_for_test_set_"+datestring+".png"),
                  dpi = 600)
      plt.close()

    plot_radar_chart(df, self.output_dir)

###############################################################################
#
#  plot histogram of test set probabilities
#
###############################################################################

    #plot histograms of probabilities  
    def plot_hist_pred_proba(y_pred_proba, directory):
      datestring = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
      plt.hist(y_pred_proba[1], bins = 20, color = "b", label = "class 1")
      plt.hist(y_pred_proba[0], bins = 20, color = "g", label = "class 0")
      plt.xlim(0, 1)
      plt.title("Histogram of predicted probabilities for class 1 in the test set")
      plt.xlabel("Predicted probability of EP_success")
      plt.ylabel("Frequency")
      plt.legend(loc = "best")
      plt.tight_layout()
      plt.savefig(os.path.join(directory, "hist_pred_proba_"+datestring+".png"), dpi = 600)
      plt.close()

    plot_hist_pred_proba(self.y_pred_proba,
                         self.output_dir)

###############################################################################
#
#  plot precision-recall curve for class 1 samples in test set
#
###############################################################################

   #plot Precision Recall Threshold curve for test set class 1
    precisions, recalls, thresholds = precision_recall_curve(self.y_test,
                                                             self.y_pred_proba_ones)

    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, directory):
      datestring = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
      plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
      plt.plot(thresholds, recalls[:-1], "g--", label = "Recall")
      plt.title("Precsion-Recall plot for classifier, test set, class 1")
      plt.xlabel("Threshold")
      plt.legend(loc = "upper left")
      plt.ylim([0, 1])
      plt.tight_layout()
      plt.savefig(os.path.join(directory, "Precision_Recall_class1_"+datestring+".png"),
                  dpi = 600)
      plt.close()

    plot_precision_recall_vs_threshold(precisions,
                                       recalls,
                                       thresholds,
                                       self.output_dir)

###############################################################################
#
#  plot ROC curve, calculate AUC and explore thresholds for class 1 samples in test set
#
###############################################################################

    #IMPORTANT: first argument is true values, second argument is predicted probabilities
    #we pass y_test and y_pred_prob
    #we do not use y_pred, because it will give incorrect results without generating an error
    #roc_curve returns 3 objects fpr, tpr, thresholds
    #fpr: false positive rate
    #tpr: true positive rate
    fpr_1, tpr_1, thresholds_1 = roc_curve(self.y_test,
                                           self.y_pred_proba_ones)
    AUC_test_class1 = round(roc_auc_score(self.y_test,
                                    self.y_pred_proba_ones), 4)
    logging.info(f"AUC score for class 1 in test set: {AUC_test_class1}")
    
    #plot ROC curves manual approach
    def plot_roc_curve(fpr, tpr, directory):
      datestring = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
      plt.plot(fpr, tpr, linewidth = 2)
      plt.plot([0, 1], [0, 1], "k--")
      plt.axis([0, 1, 0, 1])
      plt.title("ROC curve for classifier, test set, class 1") 
      plt.xlabel("False Positive Rate (1 - Specificity)")
      plt.ylabel("True Positive Rate (Sensitivity)")
      plt.grid(True)
      plt.text(0.7, 0.1, r"AUC = {AUC_test_class1}")
      plt.tight_layout()
      plt.savefig(os.path.join(directory, "ROC_curve_class1_"+datestring+".png"), dpi = 600)
      plt.close()

    plot_roc_curve(fpr_1,
                   tpr_1,
                   self.output_dir)

    #plot ROC curves using scikit_plot method
    def plot_roc_curve_skplot(y_test, y_proba, directory):
      datestring = datetime.strftime(datetime.now(), "%Y%m%d_%H%M")
      skplt.metrics.plot_roc(y_test, y_proba, title = "ROC curve")
      plt.tight_layout()
      plt.savefig(os.path.join(directory, "ROC_curve_skplt_class1_"+datestring+".png"),
                  dpi = 600)
      plt.close()

    plot_roc_curve_skplot(self.y_test,
                     self.y_pred_proba,
                     self.output_dir)

    # define a function that accepts a threshold and prints sensitivity and specificity
    def evaluate_threshold(tpr, fpr, thresholds, threshold):
      sensitivity = round(tpr[thresholds > threshold][-1], 4)
      specificity = round(1 - fpr[thresholds > threshold][-1], 4)
      
      logging.info(f"Sensitivity for class 1 at threshold {threshold}: {sensitivity}\n"
                   f"Specificity for class 1 at threshold {threshold}: {specificity}")

    evaluate_threshold(tpr_1, fpr_1, thresholds_1, 0.7)
    evaluate_threshold(tpr_1, fpr_1, thresholds_1, 0.6)
    evaluate_threshold(tpr_1, fpr_1, thresholds_1, 0.5)
    evaluate_threshold(tpr_1, fpr_1, thresholds_1, 0.4)
    evaluate_threshold(tpr_1, fpr_1, thresholds_1, 0.3)
    evaluate_threshold(tpr_1, fpr_1, thresholds_1, 0.2)

    # Try to copy log file if it was created in training.log
    try:
      shutil.copy("training.log", self.output_dir)
    except FileExistsError:
      logging.warning("Could not find training.log to copy")
    except Exception:
      logging.warning("Could not copy training.log to output directory")

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  output_dir = make_output_folder(args.outdir)
  
  ###############################################################################

  random_forest_ada_rand_search = RandomForestAdaRandSearch(metrix, output_dir)

