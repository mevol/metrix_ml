import matplotlib
matplotlib.use("Agg")

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

print("Seaborn version: ", sns.__version__)

from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime
from math import pi


# printing updates to console
def print_to_consol(message):
    print('*' *80)
    print('*    {}'.format(message))
    print('*' *80)


def get_confidence_interval(X_train, y_train, X_test, y_test, clf,
                            directory, n_iterations, kind):
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    # configure bootstrap
    n_size = int(len(X_train))

    # run bootstrap
    stats = list()
    for i in range(n_iterations):
        # prepare train and test sets
        train_boot = resample(X_train, n_samples = n_size)
        test_boot = y_train
        # fit model
        model = clf
        model.fit(train_boot, test_boot)
        # evaluate model
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        stats.append(score)

    #plot scores
    plt.hist(stats)
    plt.savefig(os.path.join(directory, 'bootstrap_hist_'+kind+date+'.png'), dpi=600)
    plt.close()
    # confidence interval
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))

    return round(alpha*100, 2), round(upper*100, 2), round(lower*100, 2)


def feature_importances_best_estimator(feature_list, directory):
      date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      #feature_list.sort(key=lambda x: x[1], reverse=True)
      feature = list(zip(*feature_list))[1]
      score = list(zip(*feature_list))[0]
      x_pos = np.arange(len(feature))
      plt.figure(figsize=(20,10))
      plt.bar(x_pos, score, align='center')
      plt.xticks(x_pos, feature, rotation=90, fontsize=12)
      plt.title('Histogram of Feature Importances for best classifer')
      plt.xlabel('Features', fontsize=12)
      plt.tight_layout()
      plt.savefig(os.path.join(directory,
                  'feature_importances_best_bar_plot_'+date+'.png'), dpi=600)
      plt.close()


def feature_importances_error_bars(clf, features, directory):
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    feature_scores = []
    for tree in clf.estimators_:
        feature_importances_ls = tree.feature_importances_
        feature_scores.append(feature_importances_ls)
        
    df = pd.DataFrame(feature_scores, columns=features)
    df_mean = df[features].mean(axis=0)
    df_std = df[features].std(axis=0)
    df_mean.plot(kind='bar', color='b', yerr=[df_std],
                 align='center', figsize=(20,10), rot=90)
    plt.title('Histogram of Feature Importances across all trees')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(directory,
                             'feature_importances_all_error_bars_'+date+'.png'), dpi=600)
    plt.close()


# feature importances by class from Gaussian Naive Bayes
def gnb_feature_importances(importance_dict, classes, directory):
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    feature = importance_dict[0].keys()
    scores = importance_dict[0].values()
    x_pos = np.arange(len(feature))
    plt.figure(figsize=(20,10))
    plt.bar(x_pos, scores, align='center')
    plt.xticks(x_pos, feature, rotation=90, fontsize=12)
    plt.title('Histogram of Feature Importances for {0}'.format(classes))
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(directory,
                'feature_importances_overall_bar_plot_'+classes+'_'+date+'.png'), dpi=600)
    plt.close()


def training_cv_stats(clf, X_train, y_train, cv):
    # accuracy for the training set
    # for each CV fold
    accuracy_each_cv = cross_val_score(clf,X_train, y_train,
                                       cv=cv, scoring='accuracy')
    # mean across all CV folds
    accuracy_mean_cv = round((cross_val_score(clf, X_train, y_train,
                                              cv=cv, scoring='accuracy').mean()) * 100 , 2)
    # calculate cross_val_scoring with different scoring functions for CV train set
    train_roc_auc = round((cross_val_score(clf, X_train, y_train,
                                           cv=cv, scoring='roc_auc').mean()) * 100, 2)
    train_recall = round((cross_val_score(clf, X_train, y_train,
                                          cv=cv, scoring='recall').mean()) * 100, 2)
    train_precision = round((cross_val_score(clf, X_train, y_train,
                                             cv=cv, scoring='precision').mean()) * 100, 2)
    train_f1 = round((cross_val_score(clf, X_train, y_train,
                                      cv=cv, scoring='f1').mean()) * 100, 2)
    # predict class and probabilities on training data with cross-validation
    y_train_CV_pred = cross_val_predict(clf, X_train, y_train, cv=cv)
    y_train_CV_pred_proba = cross_val_predict(clf, X_train, y_train,
                                              cv=cv, method='predict_proba')

    stats_dict = {'acc_cv' : accuracy_each_cv,
                  'acc' : accuracy_mean_cv,
                  'roc_auc' : train_roc_auc,
                  'recall' : train_recall,
                  'precision' : train_precision,
                  'f1-score' : train_f1}
    return stats_dict, y_train_CV_pred, y_train_CV_pred_proba

def testing_predict_stats(clf, X_test, y_test):

    #getting class predictions and probabilities on the test set for best classifiers
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)


    # calculate accuracy
    y_accuracy = round((accuracy_score(y_test, y_pred)) * 100, 2)
    # examine the class distribution of the testing set (using a Pandas Series method)
    class_dist = y_test.value_counts()
    # calculate the percentage of ones
    # because y_test only contains ones and zeros, we can simply calculate
    #the mean = percentage of ones
    ones = round(y_test.mean() * 100, 2)
    # calculate the percentage of zeros
    zeros = round((1 - y_test.mean()) * 100, 2)
    # calculate null accuracy in a single line of code
    # only for binary classification problems coded as 0/1
    null_acc = max(ones, zeros)
    # Matthews correlation coefficient
    mcc = round(matthews_corrcoef(y_test, y_pred), 4)
    stats_dict = {'predict_acc' : y_accuracy,
                  'class_distribution' : class_dist,
                  'class_one' : ones,
                  'class_zero' : zeros,
                  'null_acc' : null_acc,
                  'mcc' : mcc}
    return stats_dict, y_pred, y_pred_proba

def confusion_matrix_and_stats(y_test, y_pred, directory):
    # Plot predictions in confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    # draw confusion matrix
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    labels = ['0', '1']      
    ax = plt.subplot()
    sns.heatmap(conf_mat, annot=True, ax=ax)
    plt.title('Confusion matrix of the classifier')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(directory, 'confusion_matrix_for_test_set_'+date+'.png'), dpi=600)
    plt.close()

    # separating prediction outcomes in TP, TN, FP, FN
    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]
    
    acc_score = round(((TP + TN) / (TP + TN + FP + FN)) * 100, 2)
    class_err = round(((FP + FN) / (TP + TN + FP + FN)) * 100, 2)
    sensitivity = round((TP / (FN + TP)) * 100, 2)
    specificity = round((TN / (TN + FP)) * 100, 2)
    false_positive_rate = round((FP / (TN + FP)) * 100, 2)
    false_negative_rate = round((FN / (TP + FN)) * 100, 2)
    precision = round((TP / (TP + FP)) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)

    conf_mat_dict = {'TP' : TP,
                     'TN' : TN,
                     'FP' : FP,
                     'FN' : FN,
                     'acc' : acc_score,
                     'err' : class_err,
                     'sensitivity' : sensitivity,
                     'specificity' : specificity,
                     'FP-rate' : false_positive_rate,
                     'FN-rate' : false_negative_rate,
                     'precision' : precision,
                     'F1-score' : f1}
    return conf_mat_dict


def confusion_matrix_and_stats_3classes(y_test, y_pred, directory):
    # Plot predictions in confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred)

    # draw confusion matrix
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    labels = ['2', '1', '0']      
    ax = plt.subplot()
    sns.heatmap(conf_mat, annot=True, ax=ax)
    plt.title('Confusion matrix of the classifier')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(directory, 'confusion_matrix_for_test_set_'+date+'.png'), dpi=600)
    plt.close()

    # separating prediction outcomes in TP, TN, FP, FN
    acc_score = round(((TP + TN) / (TP + TN + FP + FN)) * 100, 2)
    class_err = round(((FP + FN) / (TP + TN + FP + FN)) * 100, 2)
    sensitivity = round((TP / (FN + TP)) * 100, 2)
    specificity = round((TN / (TN + FP)) * 100, 2)
    false_positive_rate = round((FP / (TN + FP)) * 100, 2)
    false_negative_rate = round((FN / (TP + FN)) * 100, 2)
    precision = round((TP / (TP + FP)) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)

    target_names = ['class 0', 'class 1', 'class 2']
    report = classification_report(y_true, y_pred, labels = [0, 1, 2],
                                   target_names=target_names)

    conf_mat_dict = {'TP' : TP,
                     'TN' : TN,
                     'FP' : FP,
                     'FN' : FN,
                     'acc' : acc_score,
                     'err' : class_err,
                     'sensitivity' : sensitivity,
                     'specificity' : specificity,
                     'FP-rate' : false_positive_rate,
                     'FN-rate' : false_negative_rate,
                     'precision' : precision,
                     'F1-score' : f1}
    return conf_mat_dict, report, FP, FN
    

def plot_hist_pred_proba(y_pred_proba, directory):
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    plt.hist(y_pred_proba, bins=20)
    plt.xlim(0,1)
    plt.title('Histogram of predicted probabilities for y_pred_proba to be class 1')
    plt.xlabel('Predicted probability of success')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(directory, 'hist_pred_proba_'+date+'.png'), dpi=600)
    plt.close()


#plot precision and recall curve
def plot_precision_recall_vs_threshold(y_test, y_pred_proba_ones, directory):
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_ones)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.title('Precsion-Recall plot test set class 1')
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    plt.savefig(os.path.join(directory, 'Precision_Recall_'+date+'.png'), dpi=600)
    plt.close()


#plot ROC curves
def plot_roc_curve(y_test, y_pred_proba_ones, directory):
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_ones)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.title('ROC curve for classifier on test set for class 1') 
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.savefig(os.path.join(directory, 'ROC_curve_'+date+'.png'), dpi=600)
    plt.close()
    return fpr, tpr, thresholds


# evaluate probability thresholds for sensitivity-specificity trade-off
def evaluate_threshold(tpr, fpr, thresholds):
    explore = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_dict = {}
    for i in explore:
        sensitivity = round((tpr[thresholds > i][-1]) * 100, 2)
        specificity = round((1 - fpr[thresholds > i][-1]) * 100, 2)
        threshold_dict.update({str(i) : {'sensitivity' : sensitivity,
                                        'specificity' : specificity}})
    return threshold_dict


# calibrate classifier
def calibrate_classifier(clf, X_cal, y_cal):
    clf_cccv = CalibratedClassifierCV(clf, cv='prefit')
    calibrated_clf = clf_cccv.fit(X_cal, y_cal)
    clf_acc = clf_cccv.score(X_cal, y_cal)
    return calibrated_clf, clf_acc


# radar plot for performance metrics
def plot_radar_chart(dict, directory):
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    df = pd.DataFrame.from_dict(dict, orient='index')

    # number of variable
    categories = list(df.index)
    categories = [*categories, categories[0]]

    values = df.T.loc[0].values.flatten().tolist()
    values = [*values, values[0]]
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=10)#len(values)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(label_loc, values, label='Restaurant 1')
    ax.fill(label_loc, values, "b", alpha = 0.1)
    ax.set_thetagrids(np.degrees(label_loc), labels=categories)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100%"], fontsize = 20)
    ax.set_xticklabels(categories, fontsize = 20, wrap = True)
    plt.savefig(os.path.join(directory, "radar_plot_prediction_metrics"+date+".png"), dpi=600)
    plt.close()


# plot features and coefficients for support vector machines
def plot_coefficients(coef, feature_names, directory):
    '''Once a linear SVM is fit to data (e.g., svm.fit(features, labels)), the coefficients can be accessed with svm.coef_. Recall that a linear SVM creates a hyperplane that uses support vectors to maximise the distance between the two classes. The weights obtained from svm.coef_ represent the vector coordinates which are orthogonal to the hyperplane and their direction indicates the predicted class. The absolute size of the coefficients in relation to each other can then be used to determine feature importance for the data separation task.'''
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    imp = coef
    imp,names = zip(*sorted(zip(imp, feature_names)))
    plt.barh(range(len(feature_names)), imp, align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.tight_layout()
    plt.savefig(os.path.join(directory,
                'feature_importances_coefficient_'+date+'.png'), dpi=600)
    plt.close()


# putting some code for over/under sampling here in case I ever have to use it
   #the weight distribution for the classes used by "class_weight" weights = {0:0.1, 1:0.9}
    #from imblearn.over_sampling import RandomOverSampler
    #from imblearn.under_sampling import RandomUnderSampler
    #from imblearn.over_sampling import SMOTE

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


# putting some code for visualising decision trees here in case I ever need them
# from sklearn.tree import export_graphviz
#    def visualise_tree(tree_bag, directory, columns, name):
#      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
#      trees = tree_bag.estimators_
#      i_tree = 0
#      for tree in trees:
#        with open(os.path.join(directory,'tree_clf_rand_bag_new_'+name+datestring+str(i_tree)+'.dot'), 'w') as f:
#          export_graphviz(tree, out_file=f, feature_names=columns, rounded=True, filled=True)
#          f.close()
#        dotfile = os.path.join(directory, 'tree_clf_rand_bag_new_'+name+datestring+str(i_tree)+'.dot')
#        pngfile = os.path.join(directory, 'tree_clf_rand_bag_new_'+name+datestring+str(i_tree)+'.png')
#        command = ["dot", "-Tpng", dotfile, "-o", pngfile]
#        subprocess.check_call(command)
#        i_tree = i_tree + 1
