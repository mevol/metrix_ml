###############################################################################
#
#  imports and set up environment
#
###############################################################################
'''Defining the environment for this class'''
import argparse
import pandas as pd
import os
import numpy as np
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy.stats import randint
from scipy.stats import uniform
from tbx import get_confidence_interval, feature_importances_best_estimator
from tbx import feature_importances_error_bars, confusion_matrix_and_stats_multiclass
from tbx import training_cv_stats_multiclass, testing_predict_stats_multiclass
from tbx import calibrate_classifier, plot_radar_chart, print_to_consol

def make_output_folder(outdir):
    '''A small function for making an output directory
    Args:
        outdir (str): user provided directory where the output directory will be created
        output_dir (str): the newly created output directory named
                         "randomforest_randomsearch_scaled"
    Yields:
        directory
    '''
    output_dir = os.path.join(outdir, 'randomforest_randomsearch_scaled')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class RandomForestRandSearch():
    ''' A class to conduct a randomised search and training for best parameters for a
        Random Forest for multiple classes; the following steps are executed:
        * loading input data in CSV format
        * creating output directory to write results files to
        * set up a log file to keep note of stats and processes
        * prepare the input data by splitting into a calibration (5%), testing (20%) and
          training (80%) sets and applying StandardScaler
        * conduct randomised search to find best parameters for the best predictor
        * save model to disk
        * get 95% confidence interval
        * get feature importances
        * get statistics for training using 3-fold cross-validation and testing
        * get more detailed statistics and plots for prediction performances on the testing
          set; this includes a confusion matrix, histogram of prediction probabilities,
          precision-recall curve and ROC curve
        * explore sensitivity/specificity trade-off when using different probability
          thresholds
        * calibrate the predictor and write the calibrated version to disk
        * get 95% confidence interval for calibrated classifier

        Args:

           data (str): file path to the input CSV file
           directory (str): target output directory where an output folder will be created
                            and all results will be written to
           numf (int): maximum number of features to use in training; default = 10
           numc (int): number of search cycles for randomised search; default = 500
           cv (int): number of cross-validation cycles to use during training; default = 3
           bootiter (int): number of bootstrap cylces to use for getting confidence
                           intervals; default = 1000

        Yields:
        trained predictor: "best_predictor_<date>.pkl"
        trained and calibrated predictor: "best_predictor_calibrated_<date>.pkl"
        logfile: "randomforest_randomsearch.log"
        plots: "bootstrap_hist_uncalibrated_<date>.png"
               "feature_importances_best_bar_plot_<date>.png"
               "feature_importances_all_error_bars_<date>.png"
               "confusion_matrix_for_test_set_<date>.png"
               "hist_pred_proba_<date>.png"
               "Precision_Recall_<date>.png"
               "ROC_curve_<date>.png"
               "bootstrap_hist_calibrated_<date>.png"
               "radar_plot_prediction_metrics<date>.png"
    '''
    def __init__(self, data, directory, numf, numc, cv, bootiter):
        self.numf = numf
        self.numc = numc
        self.cv = cv
        self.bootiter = bootiter
        self.data = pd.read_csv(data)
        self.directory = make_output_folder(directory)

        logging.basicConfig(level=logging.INFO, filename=os.path.join(self.directory,
                    'randomforest_randomsearch.log'), filemode='w')
        logging.info(f'Loaded input data \n'
                     f'Created output directories at {self.directory} \n')

        self.start = datetime.now()

        self.prepare_data()
        self.randomised_search()
        self.get_training_testing_prediction_stats()
        self.detailed_analysis()

###############################################################################
#
#  prepare input data
#
###############################################################################

    def prepare_data(self):
        print_to_consol('Preparing input data and split in train/test/calibration set')

        for name in self.data.columns:
            if 'success' in name or "ground_truth" in name:
                y = self.data[name]
                X = self.data.drop([name, 'Unnamed: 0'], axis=1).select_dtypes(
                                                 exclude=['object'])

        # create a 5% calibration set if needed
        X_temp, X_cal, y_temp, self.y_cal = train_test_split(X, y, test_size=0.05,
                                                        random_state=42)

        # use the remaining data for 80/20 train-test split
        X_train, X_test, self.y_train, self.y_test = train_test_split(X_temp,
                                                                        y_temp,
                                                                        test_size=0.2,
                                                                        random_state=100)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_cal_scaled = scaler.transform(X_cal)
        
        self.X_train_scaled = pd.DataFrame(data=X_train_scaled,
                              index=X_train.index,
                              columns=X_train.columns)
        self.X_test_scaled = pd.DataFrame(data=X_test_scaled,
                              index=X_test.index,
                              columns=X_test.columns)
                              
        self.X_cal_scaled = pd.DataFrame(data=X_cal_scaled,
                              index=X_cal.index,
                              columns=X_cal.columns)

        X_test_out = os.path.join(self.directory, "X_test.csv")
        np.savetxt(X_test_out, self.X_test_scaled, delimiter=",")

        y_test_out = os.path.join(self.directory, "y_test.csv")
        np.savetxt(y_test_out, self.y_test, delimiter=",")

        logging.info(f'Writing X_test and y_test to disk \n')
        logging.info(f'Created test, train and validation set \n'
                     f'Scaling the train set and applying to test set and calibration set \n')

###############################################################################
#
#  randomized search
#
###############################################################################

    def randomised_search(self):
        print_to_consol('Running randomized search to find best classifier')

        #create the decision forest
        clf1 = RandomForestClassifier(random_state=20,
                                      class_weight='balanced',
                                      max_features = self.numf)

        logging.info(f'Initialised classifier \n')

        #set up randomized search
        param_dict = {
                   'criterion': ['gini', 'entropy'],
                   'n_estimators': randint(100, 10000),#number of base estimators to use
                   'min_samples_split': randint(2, 20),
                   'max_depth': randint(1, 10),
                   'min_samples_leaf': randint(1, 20),
                   'max_leaf_nodes': randint(10, 20)}

        logging.info(f'Following parameters will be explored in randomized search \n'
                     f'{param_dict} \n')

        #building and running the randomized search
        rand_search = RandomizedSearchCV(clf1, param_dict, random_state=5,
                                         cv=self.cv, n_iter=self.numc,
                                         scoring='accuracy', n_jobs=-1)

        rand_search_fitted = rand_search.fit(self.X_train_scaled,
                                             self.y_train)
                                             
        best_parameters = rand_search_fitted.best_params_
        best_scores = rand_search_fitted.best_score_

        logging.info(f'Running randomised search for best patameters of classifier \n'
                     f'Best parameters found: {best_parameters} \n'
                     f'Best accuracy scores found: {best_scores} \n')
                     
        self.model = rand_search_fitted.best_estimator_

        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        joblib.dump(self.model, os.path.join(self.directory,
                    'best_predictor_'+datestring+'.pkl'))

        logging.info(f'Writing best classifier to disk in {self.directory} \n')

        print_to_consol('Getting 95% confidence interval for uncalibrated classifier')

        alpha, upper, lower = get_confidence_interval(self.X_train_scaled, self.y_train,
                                                      self.X_test_scaled, self.y_test,
                                                      self.model, self.directory,
                                                      self.bootiter, 'uncalibrated')

        logging.info(f'{alpha}% confidence interval {upper}% and {lower}% \n'
                     f'for uncalibrated classifier. \n')

        print_to_consol('Getting feature importances for best classifier')

        best_clf_feat_import = self.model.feature_importances_
        best_clf_feat_import_sorted = sorted(zip(best_clf_feat_import,
                                                self.X_train_scaled.columns),
                                                reverse=True)

        logging.info(f'Feature importances for best classifier {best_clf_feat_import_sorted} \n')

        all_clf_feat_import_mean = np.mean(
                 [tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        all_clf_feat_import_mean_sorted = sorted(zip(all_clf_feat_import_mean,
                                                self.X_train_scaled.columns),
                                                reverse=True)

        print_to_consol('Plotting feature importances for best classifier')

        feature_importances_best_estimator(best_clf_feat_import_sorted, self.directory)
        logging.info(f'Plotting feature importances for best classifier in decreasing order \n')
        feature_importances_error_bars(self.model, self.X_train_scaled.columns, self.directory)
        logging.info(f'Plotting feature importances for best classifier with errorbars \n')

###############################################################################
#
#  get training and testing stats
#
###############################################################################

    def get_training_testing_prediction_stats(self):
        print_to_consol('Getting basic stats for training set and cross-validation')

        training_stats, y_train_pred, y_train_pred_proba = training_cv_stats_multiclass(
                                                self.model, self.X_train_scaled,
                                                self.y_train, self.cv)

        logging.info(f'Basic stats achieved for training set and 3-fold CV \n'
            f'Accuracy for each individual fold of 3 CV folds: {training_stats["acc_cv"]} \n'
            f'Accuracy across all 3 CV-folds: {training_stats["acc"]} \n'
            f'Recall across all 3 CV-folds: {training_stats["recall"]} \n'
            f'Precision across all 3 CV-folds: {training_stats["precision"]} \n'
            f'F1 score across all 3 CV-folds: {training_stats["f1-score"]} \n'
            f'Storing cross-validated y_train classes in y_train_pred \n'
            f'Storing cross-validated y_train probabilities in y_train_pred_proba \n')

        print_to_consol('Getting class predictions and probabilities for test set')

        test_stats, self.y_pred, self.y_pred_proba = testing_predict_stats_multiclass(
                                                self.model, self.X_test_scaled, self.y_test)

        logging.info(f'Predicting on the test set. \n'
                     f'Storing classes in y_pred and probabilities in y_pred_proba \n')

        print_to_consol(
            'Calculate prediction stats for y_pred and y_pred_proba of test set')

        logging.info(f'Basic stats on the test set. \n'
                     f'Prediction accuracy on the test set: {test_stats["predict_acc"]} \n'
                     f'Class distributio in the test set: {test_stats["class_distribution"]} \n'
                     f'Matthews Correlation Coefficient: {test_stats["mcc"]} \n')

###############################################################################
#
#  get more detailed stats and plots
#
###############################################################################

    def detailed_analysis(self):
        print_to_consol('Making a confusion matrix for test set classification outcomes')

        matrix_stats, report = confusion_matrix_and_stats_multiclass(self.y_test, self.y_pred,
                                                  'before_cal', self.directory)

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

        logging.info(f'Classification report on test set before calibration. \n'
                     f'{report} \n')

        print_to_consol('Make a radar plot for performance metrics')

        radar_dict = {'Classification accuracy' : matrix_stats["acc"],
                      'Classification error' : matrix_stats["err"],
                      'Sensitivity' : matrix_stats["sensitivity"],
                      'Specificity' : matrix_stats["specificity"],
                      'False positive rate' : matrix_stats["FP-rate"],
                      'False negative rate' : matrix_stats["FN-rate"],
                      'Precision' : matrix_stats["precision"],
                      'F1-score' : matrix_stats["F1-score"],
                      'ROC AUC' : None}

        plot_radar_chart(radar_dict, self.directory)

        print_to_consol(
            'Calibrating classifier and writing to disk; getting new accuracy')

        self.calibrated_clf, clf_acc = calibrate_classifier(self.model, self.X_cal_scaled,
                                                            self.y_cal)

        date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        joblib.dump(self.calibrated_clf, os.path.join(self.directory,
                    'best_calibrated_predictor_'+date+'.pkl'))

        logging.info(
          f'Calibrated the best classifier with X_cal and y_cal and new accuracy {clf_acc}\n'
          f'Writing file to disk disk in {self.directory} \n')

        print_to_consol('Getting 95% confidence interval for calibrated classifier')

        alpha, upper, lower = get_confidence_interval(self.X_train_scaled, self.y_train,
                                                      self.X_test_scaled, self.y_test,
                                                      self.calibrated_clf, self.directory,
                                                      self.bootiter, 'calibrated')

        logging.info(f'{alpha}% confidence interval {upper}% and {lower}% \n'
                     f'for calibrated classifier. \n')

        print_to_consol('Running prediction for calibrated classifier')

        print_to_consol(
        'Getting class predictions and probabilities for test set with calibrated classifier')

        test_stats_cal, self.y_pred_cal, self.y_pred_proba_cal = testing_predict_stats_multiclass(
                                                self.calibrated_clf,
                                                self.X_test_scaled, self.y_test)

        logging.info(
        f'Predicting on the test set with calibrated classifier. \n'
        f'Storing classes for calibrated classifier in y_pred and probabilities in y_pred_proba. \n')

        print_to_consol(
        'Calculate prediction stats for y_pred and y_pred_proba of test set with calibrated classifier')

        logging.info(f'Basic stats on the test set woth calibrated classifier. \n'
                     f'Prediction accuracy on the test set: {test_stats_cal["predict_acc"]} \n'
                     f'Class distributio in the test set: {test_stats_cal["class_distribution"]} \n'
                     f'Matthews Correlation Coefficient: {test_stats_cal["mcc"]} \n')

        print_to_consol(
        'Making a confusion matrix for test set classification outcomes with calibrated classifier')

        matrix_stats_cal, report_cal = confusion_matrix_and_stats_multiclass(self.y_test, self.y_pred_cal,
                                                  'after_cal', self.directory)

        logging.info(f'Detailed analysis of confusion matrix for test set with calibrated classifier. \n'
                     f'True positives: {matrix_stats_cal["TP"]} \n'
                     f'True negatives: {matrix_stats_cal["TN"]} \n'
                     f'False positives: {matrix_stats_cal["FP"]} \n'
                     f'False negatives: {matrix_stats_cal["FN"]} \n'
                     f'Classification accuracy: {matrix_stats_cal["acc"]} \n'
                     f'Classification error: {matrix_stats_cal["err"]} \n'
                     f'Sensitivity: {matrix_stats_cal["sensitivity"]} \n'
                     f'Specificity: {matrix_stats_cal["specificity"]} \n'
                     f'False positive rate: {matrix_stats_cal["FP-rate"]} \n'
                     f'False negative rate: {matrix_stats_cal["FN-rate"]} \n'
                     f'Precision: {matrix_stats_cal["precision"]} \n'
                     f'F1-score: {matrix_stats_cal["F1-score"]} \n')

        logging.info(f'Classification report on test set afetr callibration. \n'
                     f'{report_cal} \n')

        print_to_consol('Make a radar plot for performance metrics with calibrated classifier')

        radar_dict_cal = {'Classification accuracy' : matrix_stats_cal["acc"],
                      'Classification error' : matrix_stats_cal["err"],
                      'Sensitivity' : matrix_stats_cal["sensitivity"],
                      'Specificity' : matrix_stats_cal["specificity"],
                      'False positive rate' : matrix_stats_cal["FP-rate"],
                      'False negative rate' : matrix_stats_cal["FN-rate"],
                      'Precision' : matrix_stats_cal["precision"],
                      'F1-score' : matrix_stats_cal["F1-score"],
                      'ROC AUC' : None}

        plot_radar_chart(radar_dict_cal, self.directory)

        end = datetime.now()
        duration = end - self.start

        logging.info(f'Training lasted for {duration} minutes \n')

        logging.info(f'Training completed \n')

        print_to_consol('Training completed')


def run(input_csv, output_dir, features, cycles, boot_iter, cv):

    RandomForestRandSearch(input_csv, output_dir, features, cycles, boot_iter, cv)



def main():
    '''defining the command line input to make it runable'''
    parser = argparse.ArgumentParser(description='Random Forest randomized search')

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
        '--cv',
        type=int,
        dest='cv',
        default=3,
        help='Number of cross-validation repeats to use during training')

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
        args.cv,
        args.boot_iter)


if __name__ == "__main__":
    main()
