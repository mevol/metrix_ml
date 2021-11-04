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
from datetime import datetime
from tbx import  confusion_matrix_and_stats, plot_radar_chart
from tbx import testing_predict_stats, plot_hist_pred_proba, print_to_consol
from tbx import plot_precision_recall_vs_threshold, plot_roc_curve, evaluate_threshold

def make_output_folder(outdir):
    '''A small function for making an output directory
    Args:
        outdir (str): user provided directory where the output directory will be created
        output_dir (str): the newly created output directory named
                         "predict"
    Yields:
        directory
    '''
    output_dir = os.path.join(outdir, 'predict')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class PredictWithSplit():
    ''' A class to run predcit step on any model with any data; this is because I forgot
        this step in the actual training script
          the following steps are executed:
        * loading input data in CSV format
        * creating output directory to write results files to
        * set up a log file to keep note of stats and processes
        * prepare the input data by splitting into a calibration (5%), testing (20%) and
          training (80%) sets
        * run predict
        * get stats for test set

        Args:

           data (str): file path to the input CSV file
           directory (str): target output directory where an output folder will be created
                            and all results will be written to
           model (str): file path to the model

        Yields:
        trained predictor: "best_predictor_<date>.pkl"
        trained and calibrated predictor: "best_predictor_calibrated_<date>.pkl"
        logfile: "decisiontree_randomsearch.log"
        plots: "confusion_matrix_for_test_set_<date>.png"
               "hist_pred_proba_<date>.png"
               "Precision_Recall_<date>.png"
               "ROC_curve_<date>.png"
               "radar_plot_prediction_metrics<date>.png"
    '''
    def __init__(self, data, directory, model):
        with open(model, 'rb') as f:
            self.model = joblib.load(f)
        self.data = pd.read_csv(data)
        self.directory = make_output_folder(directory)

        logging.basicConfig(level=logging.INFO, filename=os.path.join(self.directory,
                    'predict.log'), filemode='w')
        logging.info(f'Loaded input data \n'
                     f'Created output directories at {self.directory} \n')

        self.start = datetime.now()

        self.prepare_data()
        self.predict()
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
        X_temp, self.X_cal, y_temp, self.y_cal = train_test_split(X, y, test_size=0.05,
                                                        random_state=42)

        # use the remaining data for 80/20 train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_temp,
                                                                        y_temp,
                                                                        test_size=0.2,
                                                                        random_state=100)

        logging.info(f'Created test, train and validation set \n')

###############################################################################
#
#  predict
#
###############################################################################

    def predict(self):

        print_to_consol('Running prediction for provided classifier')

        print_to_consol('Getting class predictions and probabilities for test set')

        test_stats, self.y_pred, self.y_pred_proba = testing_predict_stats(
                                                self.model, self.X_test, self.y_test)

        logging.info(f'Predicting on the test set. \n'
                     f'Storing classes in y_pred and probabilities in y_pred_proba \n')

        print_to_consol(
            'Calculate prediction stats for y_pred and y_pred_proba of test set')

        logging.info(f'Basic stats on the test set. \n'
                     f'Prediction accuracy on the test set: {test_stats["predict_acc"]} \n'
                     f'Class distributio in the test set: {test_stats["class_distribution"]} \n'
                     f'Matthews Correlation Coefficient: {test_stats["mcc"]} \n'
                     f'Average number of class 1 samples: {test_stats["class_one"]} \n'
                     f'Average number of class 0 samples: {test_stats["class_zero"]} \n'
                     f'Null accuracy: {test_stats["null_acc"]} \n')

        print_to_consol(
            'Plotting histogram for class 1 prediction probabilities for test set')

        #store the predicted probabilities for class 1 of test set
        self.y_pred_proba_ones = self.y_pred_proba[:, 1]

        plot_hist_pred_proba(self.y_pred_proba_ones, self.directory)

        logging.info(
          f'Plotting prediction probabilities for class 1 in test set in histogram. \n')

###############################################################################
#
#  get more detailed stats and plots
#
###############################################################################

    def detailed_analysis(self):
        print_to_consol('Making a confusion matrix for test set classification outcomes')

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

        print_to_consol(
                    'Plotting precision recall curve for test set class 1 probabilities')

        logging.info(
          f'Plotting precision recall curve for class 1 in test set probabilities. \n')
        
        plot_precision_recall_vs_threshold(self.y_test, self.y_pred_proba_ones,
                                           self.directory)

        print_to_consol(
              'Plotting ROC curve ad calculating AUC for test set class 1 probabilities')

        logging.info(
          f'Plotting ROC curve for class 1 in test set probabilities. \n')

        self.fpr, self.tpr, self.thresholds = plot_roc_curve(self.y_test,
                                                   self.y_pred_proba_ones, self.directory)

        AUC = round(roc_auc_score(self.y_test, self.y_pred_proba_ones) * 100, 2)

        logging.info(
          f'Calculating AUC for ROC curve for class 1 in test set probabilities: {AUC} \n')

        print_to_consol('Make a radar plot for performance metrics')

        radar_dict = {'Classification accuracy' : matrix_stats["acc"],
                      'Classification error' : matrix_stats["err"],
                      'Sensitivity' : matrix_stats["sensitivity"],
                      'Specificity' : matrix_stats["specificity"],
                      'False positive rate' : matrix_stats["FP-rate"],
                      'False negative rate' : matrix_stats["FN-rate"],
                      'Precision' : matrix_stats["precision"],
                      'F1-score' : matrix_stats["F1-score"],
                      'ROC AUC' : AUC}

        plot_radar_chart(radar_dict, self.directory)

        print_to_consol(
            'Exploring probability thresholds, sensitivity, specificity for class 1')

        threshold_dict = evaluate_threshold(self.tpr, self.fpr, self.thresholds)

        logging.info(
          f'Exploring different probability thresholds and sensitivity-specificity trade-offs. \n'
          f'Threshold 0.2: {threshold_dict["0.2"]} \n'
          f'Threshold 0.3: {threshold_dict["0.3"]} \n'
          f'Threshold 0.4: {threshold_dict["0.4"]} \n'
          f'Threshold 0.5: {threshold_dict["0.5"]} \n'
          f'Threshold 0.6: {threshold_dict["0.6"]} \n'
          f'Threshold 0.7: {threshold_dict["0.7"]} \n'
          f'Threshold 0.8: {threshold_dict["0.8"]} \n'
          f'Threshold 0.9: {threshold_dict["0.9"]} \n')

        end = datetime.now()
        duration = end - self.start

        logging.info(f'Prediction and analysis lasted for {duration} minutes \n')

        logging.info(f'Prediction and analysis completed \n')

        print_to_consol('Prediction and analysis completed')


def run(input_csv, output_dir, model):

    PredictWithSplit(input_csv, output_dir, model)



def main():
    '''defining the command line input to make it runable'''
    parser = argparse.ArgumentParser(
                           description='prediction without standardisation/normalisation')

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
        '--model',
        type=str,
        dest='model',
        default='',
        help='Number of features to look for')

    args = parser.parse_args()

    if args.input == '':
        parser.print_help()
        exit(0)

    run(args.input,
        args.outdir,
        args.model)


if __name__ == "__main__":
    main()
