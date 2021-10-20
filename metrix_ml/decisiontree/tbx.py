import os
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime


def get_confidence_interval(X_train, y_train, X_test, y_test, clf,
                            directory, n_iterations):
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
    plt.savefig(os.path.join(directory, 'bootstrap_hist'+date+'.png'))
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
                  'feature_importances_best_bar_plot_rand_ada_'+date+'.png'))
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
                             'feature_importances_all_error_bars_'+date+'.png'))
    plt.close()

def training_cv_stats(clf, X_train, y_train):
    # accuracy for the training set
    # for each CV fold
    accuracy_each_cv = cross_val_score(clf,X_train, y_train,
                                       cv=2, scoring='accuracy')
    # mean across all CV folds
    accuracy_mean_cv = round((cross_val_score(clf, X_train, y_train,
                                              cv=2, scoring='accuracy').mean()) * 100 , 2)
    # calculate cross_val_scoring with different scoring functions for CV train set
    train_roc_auc = round((cross_val_score(clf, X_train, y_train,
                                           cv=2, scoring='roc_auc').mean()) * 100, 2)
    train_recall = round((cross_val_score(clf, X_train, y_train,
                                          cv=2, scoring='recall').mean()) * 100, 2)
    train_precision = round((cross_val_score(clf, X_train, y_train,
                                             cv=2, scoring='precision').mean()) * 100, 2)
    train_f1 = round((cross_val_score(clf, X_train, y_train,
                                      cv=2, scoring='f1').mean()) * 100, 2)
    # predict class and probabilities on training data with cross-validation
    y_train_CV_pred = cross_val_predict(clf, X_train, y_train, cv=2)
    y_train_CV_pred_proba = cross_val_predict(clf, X_train, y_train,
                                              cv=2, method='predict_proba')

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
    plt.savefig(os.path.join(directory, 'confusion_matrix_for_test_set_'+date+'.png'))
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


def plot_hist_pred_proba(y_pred_proba, directory):
    date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    plt.hist(y_pred_proba, bins=20)
    plt.xlim(0,1)
    plt.title('Histogram of predicted probabilities for y_pred_proba to be class 1')
    plt.xlabel('Predicted probability of success')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(directory, 'hist_pred_proba_'+date+'.png'))
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
    plt.savefig(os.path.join(directory, 'Precision_Recall_'+date+'.png'))
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
    plt.savefig(os.path.join(directory, 'ROC_curve_'+date+'.png'))
    plt.close()
        
 
      





