def run():
	###############################################################################
	#
	#  importing packages
	#
	###############################################################################

	import pandas as pd
	import os
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn.model_selection import train_test_split

	###############################################################################
	#
	#  importing database table and loading as pandas dataframe
	#
	###############################################################################

	#import data which is held in a CSV file previously exported from METRIX_database and edited by hand
	METRIX_PATH = "/Users/melanievollmar/Documents/METRICS/database_output_analysis/metrix_db_20170531/Python_ML/data"
	def load_metrix_data(metrix_path = METRIX_PATH):
			csv_path = os.path.join(metrix_path, "May_2017_combined_valid_results_EP-Shelx_fail_removed.csv")
			return pd.read_csv(csv_path)

	#look at the imported data to get an idea what we are working with
	metrix = load_metrix_data()

	###############################################################################
	#
	#  creating 3 data frames specific to the three development milestones I had
	#  1--> directly from data processing
	#  2--> after adding protein information
	#  3--> carrying out some further column transformations
	#
	###############################################################################

	#look at the data that is coming from processing
	attr_database = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
									 'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
									 'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
									 'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF']
	metrix_database = metrix[attr_database]

	#database plus manually added data
	attr_man_add = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
									'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
									'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
									'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF',
									'wavelength', 'Vcell', 'Matth_coeff', 'No_atom_chain', 'solvent_content',
									'No_mol_ASU', 'MW_chain', 'sites_ASU']
	metrix_man_add = metrix[attr_man_add]

	#after column transformation expected feature list
	#X_transform_train_ordered = X_transform_train[['IoverSigma', 'cchalf', 'RmergediffI', 'RmergeI', 'RmeasI',
	#                          'RmeasdiffI', 'RpimdiffI', 'RpimI', 'totalobservations',
	#                          'totalunique', 'multiplicity', 'completeness', 'lowreslimit',
	#                          'highreslimit', 'wilsonbfactor', 'anomalousslope',
	#                          'anomalousCC', 'anomalousmulti', 'anomalouscompl', 'diffI',
	#                          'diffF', 'wavelength', 'wavelength**3', 'wavelength**3/Vcell',
	#                          'Vcell', 'solvent_content', 'Vcell/Vm<Ma>', 'Matth_coeff',
	#                          'MW_ASU/sites_ASU/solvent_content', 'MW_chain', 'No_atom_chain',
	#                          'No_mol_ASU', 'MW_ASU', 'sites_ASU', 'MW_ASU/sites_ASU',
	#                          'MW_chain/No_atom_chain', 'wilson', 'bragg', 'volume_wilsonB_highres']]                              

	metrix_transform = metrix_man_add.copy()

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

	###############################################################################
	#
	#  creating training and test set for each of the 3 dataframes
	#
	###############################################################################

	X_database = metrix_database
	X_man_add = metrix_man_add
	X_transform = metrix_transform
	y = metrix['EP_success']

	X_database_train, X_database_test, y_train, y_test = train_test_split(X_database, y, test_size=0.2, random_state=42)
	X_man_add_train, X_man_add_test, y_train, y_test = train_test_split(X_man_add, y, test_size=0.2, random_state=42)
	X_transform_train, X_transform_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.2, random_state=42)

	#print(X_database.columns, X_database.shape)
	#print(X_man_add.columns, X_man_add.shape)
	#print(X_transform.columns, X_transform.shape)

	#print(X_database_train.columns, X_database_train.shape)
	#print(X_man_add_train.columns, X_man_add_train.shape)
	#print(X_transform_train.columns, X_transform_train.shape)

	assert X_database.columns.all() == X_database_train.columns.all()
	assert X_man_add.columns.all() == X_man_add_train.columns.all()
	assert X_transform.columns.all() == X_transform_train.columns.all()

	###############################################################################
	#
	#  a basic desicion tree classifier with cross_validation
	#
	###############################################################################
	#training a decision tree with the prepared train set and train set lables

	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import mean_squared_error
	from sklearn.externals import joblib
	from sklearn.model_selection import GridSearchCV
	from sklearn.model_selection import cross_val_score
	from sklearn.model_selection import cross_val_predict
	from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
	from sklearn.metrics import precision_recall_curve, roc_curve
	from sklearn.tree import export_graphviz

	#create the decision tree
	tree_clf = DecisionTreeClassifier(random_state=42)

	#set up grid search
	param_grid = {"criterion": ["gini", "entropy"],
								'max_features': [1, 2, 4, 8, 16],
								 "min_samples_split": [5, 10, 15], #min samples per node to induce split
								 "max_depth": [3, 4, 5, 6], #max number of splits to do
								 "min_samples_leaf": [2, 4, 6], #min number of samples in a leaf
								 "max_leaf_nodes": [5, 10, 15]}#max number of leaves
						 

	#building and running the grid search
	grid_search = GridSearchCV(tree_clf, param_grid, cv=10,
														scoring='accuracy')

	grid_search.fit(X_transform_train, y_train)

	#get best parameter combination and its score as accuracy
	print(grid_search.best_params_)
	print(grid_search.best_score_)
	feature_importances = grid_search.best_estimator_.feature_importances_
	print(sorted(zip(feature_importances, X_transform_train), reverse=True))
	
	
	###############################################################################
	#
	#  create tree with best parameters
	#
	###############################################################################
	#create a new tree with the best settings found above
	tree_clf_new = DecisionTreeClassifier(max_depth=5,
																				max_leaf_nodes=15,
																				min_samples_leaf= 2,
																				min_samples_split=5,
																				max_features=16,
																				random_state= 42)

	#visualise best decision tree
	import subprocess
	tree_clf_new.fit(X_transform_train, y_train)
	dotfile = os.path.join(METRIX_PATH, 'tree_clf_new.dot')
	pngfile = os.path.join(METRIX_PATH, 'tree_clf_new.png')

	with open(dotfile, 'w') as f:
			export_graphviz(tree_clf_new, out_file=f, feature_names=X_transform_train.columns,
										 rounded=True, filled=True)
												
	command = ["dot", "-Tpng", dotfile, "-o", pngfile]
	subprocess.check_call(command)


	#not the best measure to use as it heavily depends on the sample 
	#distribution --> accuracy
	print(cross_val_score(tree_clf_new, X_transform_train, y_train,
									cv=10, scoring='accuracy'))
	print(cross_val_score(tree_clf_new, X_transform_train, y_train,
									cv=10, scoring='accuracy').mean())



	# calculate cross_val_scoring with different scoring functions for CV train set
	from sklearn.model_selection import cross_val_score
	train_roc_auc = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='roc_auc').mean()
	train_accuracy = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='accuracy').mean()
	train_recall = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='recall').mean()
	train_precision = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='precision').mean()
	train_f1 = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='f1').mean()

	print('ROC_AUC CV', train_roc_auc)#uses metrics.roc_auc_score
	print('Accuracy CV', train_accuracy)#uses metrics.accuracy_score
	print('Recall CV', train_recall)#uses metrics.recall_score
	print('Precision CV', train_precision)#uses metrics.precision_score
	print('F1 score CV', train_f1)#uses metrics.f1_score

	###############################################################################
	#
	#  general analysis to find how good the predictions are
	#
	###############################################################################

	#try out how well the classifier works to predict from the test set
	y_pred_class = tree_clf_new.predict(X_transform_test)

	#alternative way to not have to use the test set
	y_train_pred = cross_val_predict(tree_clf_new, X_transform_train, y_train,
									cv=10)

	# calculate accuracy
	from sklearn import metrics
	print(metrics.accuracy_score(y_test, y_pred_class))

	# examine the class distribution of the testing set (using a Pandas Series method)
	print(y_test.value_counts())

	# calculate the percentage of ones
	# because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
	print(y_test.mean())

	# calculate the percentage of zeros
	print(1 - y_test.mean())

	# calculate null accuracy in a single line of code
	# only for binary classification problems coded as 0/1
	max(y_test.mean(), 1 - y_test.mean())

	# calculate null accuracy (for multi-class classification problems)
	print(y_test.value_counts().head(1) / len(y_test))

	# print the first 25 true and predicted responses
	print('True:', y_test.values[0:25])
	print('False:', y_pred_class[0:25])#comes from the testset
	print('False:', y_train_pred[0:25])#comes from the cross-validated trainset

	###############################################################################
	#
	#  improved analysis by using a confusion matrix
	#
	###############################################################################

	# IMPORTANT: first argument is true values, second argument is predicted values
	# this produces a 2x2 numpy array (matrix)
	print('confusion matrix using test set', metrics.confusion_matrix(y_test, y_pred_class))#on the test set
	print('confusion matrix using CV train set', metrics.confusion_matrix(y_train, y_train_pred))#on the CV train set

	# save confusion matrix and slice into four pieces
	confusion = metrics.confusion_matrix(y_test, y_pred_class)
	confusion_CV = metrics.confusion_matrix(y_train, y_train_pred)
	#print(confusion)
	#print(confusion_CV)
	#[row, column] for test set
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]

	#[row, column] for CV train set
	TP_CV = confusion_CV[1, 1]
	TN_CV = confusion_CV[0, 0]
	FP_CV = confusion_CV[0, 1]
	FN_CV = confusion_CV[1, 0]

	#metrics calculated from confusion matrix
	# use float to perform true division, not integer division
	print('accuracy score manual', (TP + TN) / float(TP + TN + FP + FN))
	print('accuracy score sklearn', metrics.accuracy_score(y_test, y_pred_class))
	print('accuracy score manual CV', (TP_CV + TN_CV) / float(TP_CV + TN_CV + FP_CV + FN_CV))
	print('accuracy score sklearn CV', metrics.accuracy_score(y_train, y_train_pred))

	#something of one class put into the other
	classification_error = (FP + FN) / float(TP + TN + FP + FN)
	print('classification error manual', classification_error)
	print('classification error sklearn', 1 - metrics.accuracy_score(y_test, y_pred_class))
	classification_error_CV = (FP_CV + FN_CV) / float(TP_CV + TN_CV + FP_CV + FN_CV)
	print('classification error manual CV', classification_error_CV)
	print('classification error sklearn CV', 1 - metrics.accuracy_score(y_train, y_train_pred))

	#same as recall or true positive rate; correctly placed positive cases
	sensitivity = TP / float(FN + TP)
	print('sensitivity manual', sensitivity)
	print('sensitivity sklearn', metrics.recall_score(y_test, y_pred_class))
	sensitivity_CV = TP_CV / float(FN_CV + TP_CV)
	print('sensitivity manual CV', sensitivity_CV)
	print('sensitivity sklearn CV', metrics.recall_score(y_train, y_train_pred))

	specificity = TN / (TN + FP)
	print('specificity manual', specificity)
	specificity_CV = TN_CV / (TN_CV + FP_CV)
	print('specificity manual CV', specificity_CV)

	false_positive_rate = FP / float(TN + FP)
	print('false positive rate manual', false_positive_rate)
	print('false positive rate manual alternative', 1 - specificity)
	false_positive_rate_CV = FP_CV / float(TN_CV + FP_CV)
	print('false positive rate manual CV', false_positive_rate_CV)
	print('false positive rate manual alternative CV', 1 - specificity_CV)

	#how confidently the correct placement was done
	precision = TP / float(TP + FP)
	print('precision manual', precision)
	print('precision sklearn', metrics.precision_score(y_test, y_pred_class))
	precision_CV = TP_CV / float(TP_CV + FP_CV)
	print('precision manual CV', precision_CV)
	print('precision sklearn CV', metrics.precision_score(y_train, y_train_pred))

	#F1 score; uses precision and recall
	from sklearn.metrics import f1_score
	print('F1 score', f1_score(y_test, y_pred_class))
	print('F1 score CV', f1_score(y_train, y_train_pred))

	###############################################################################
	#
	#  adjusting classification thresholds; default threshold is 0.5
	#
	###############################################################################

	# print the first 10 predicted responses
	# 1D array (vector) of binary values (0, 1)
	tree_clf_new.predict(X_transform_test)[0:10]#test set

	# print the first 10 predicted probabilities of class membership
	print(tree_clf_new.predict_proba(X_transform_test)[0:10])#test set

	#probabilities for the CV train set
	y_scores=tree_clf_new.predict_proba(X_transform_train)#train set
	print(y_scores[0:10])

	# print the first 10 predicted probabilities for class 1
	print(tree_clf_new.predict_proba(X_transform_test)[0:10, 1])#test set

	# store the predicted probabilities for class 1
	y_pred_prob = tree_clf_new.predict_proba(X_transform_test)[:, 1]#test set

	# histogram of predicted probabilities

	# 8 bins for prediction probability on the test set
	plt.hist(y_pred_prob, bins=8)
	# x-axis limit from 0 to 1
	plt.xlim(0,1)
	plt.title('Histogram of predicted probabilities')
	plt.xlabel('Predicted probability of EP_success')
	plt.ylabel('Frequency')
	plt.show()

	# 8 bins for prediction probability on the CV train set
	plt.hist(y_scores[:,1], bins=8)
	# x-axis limit from 0 to 1
	plt.xlim(0,1)
	plt.title('Histogram of predicted probabilities')
	plt.xlabel('Predicted probability of EP_success')
	plt.ylabel('Frequency')
	plt.show()

	###############################################################################
	#
	#  adjusting thresholds using precision vs recall curve
	#
	###############################################################################

	#plot Precision Recall Threshold curve for test set 
	precisions, recalls, thresholds_forest = precision_recall_curve(y_test, y_pred_prob)

	def plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest):
			plt.plot(thresholds_forest, precisions[:-1], "b--", label="Precision")
			plt.plot(thresholds_forest, recalls[:-1], "g--", label="Recall")
			plt.xlabel("Threshold")
			plt.legend(loc="upper left")
			plt.ylim([0,1])

	plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest)
	plt.show()

	#plot Precision Recall Threshold curve for CV train set 
	precisions, recalls, thresholds_forest = precision_recall_curve(y_train, y_scores[:,1])

	def plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest):
			plt.plot(thresholds_forest, precisions[:-1], "b--", label="Precision")
			plt.plot(thresholds_forest, recalls[:-1], "g--", label="Recall")
			plt.xlabel("Threshold")
			plt.legend(loc="upper left")
			plt.ylim([0,1])

	plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest)
	plt.show()

	###############################################################################
	#
	#  looking at classifier performance using ROC curves and
	#  adjusting thresholds
	#
	###############################################################################

	# IMPORTANT: first argument is true values, second argument is predicted probabilities

	# we pass y_test and y_pred_prob
	# we do not use y_pred_class, because it will give incorrect results without generating an error
	# roc_curve returns 3 objects fpr, tpr, thresholds
	# fpr: false positive rate
	# tpr: true positive rate
	fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)#test set

	plt.plot(fpr, tpr)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.rcParams['font.size'] = 12
	plt.title('ROC curve for EP_success classifier')
	plt.xlabel('False Positive Rate (1 - Specificity)')
	plt.ylabel('True Positive Rate (Sensitivity)')
	plt.grid(True)
	plt.show()

	#as found in sklearn package
	fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)#test set

	def plot_roc_curve(fpr, tpr, label=None):
			plt.plot(fpr, tpr, linewidth=2, label=label)
			plt.plot([0, 1], [0, 1], 'k--')
			plt.axis([0, 1, 0, 1])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
		
	plot_roc_curve(fpr, tpr)
	plt.show()

	#for the CV training set
	fpr_CV, tpr_CV, thresholds_CV = metrics.roc_curve(y_train, y_scores[:,1])#CV train set

	plt.plot(fpr_CV, tpr_CV)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.rcParams['font.size'] = 12
	plt.title('ROC curve for EP_success classifier')
	plt.xlabel('False Positive Rate (1 - Specificity)')
	plt.ylabel('True Positive Rate (Sensitivity)')
	plt.grid(True)
	plt.show()

	#as found in sklearn package
	fpr_CV, tpr_CV, thresholds_CV = roc_curve(y_train, y_scores[:,1])#CV train set

	def plot_roc_curve(fpr_CV, tpr_CV, label=None):
			plt.plot(fpr_CV, tpr_CV, linewidth=2, label=label)
			plt.plot([0, 1], [0, 1], 'k--')
			plt.axis([0, 1, 0, 1])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
		
	plot_roc_curve(fpr_CV, tpr_CV)
	plt.show()

	# define a function that accepts a threshold and prints sensitivity and specificity
	def evaluate_threshold(threshold):
			print('Sensitivity:', tpr[thresholds > threshold][-1])
			print('Specificity:', 1 - fpr[thresholds > threshold][-1])
		
	print('Threshold 0.5', evaluate_threshold(0.5))
	print('Threshold 0.4', evaluate_threshold(0.4))

	def evaluate_threshold_CV(threshold_CV):
			print('Sensitivity:', tpr_CV[thresholds_CV > threshold_CV][-1])
			print('Specificity:', 1 - fpr_CV[thresholds_CV > threshold_CV][-1])
		
	print('Threshold CV 0.5', evaluate_threshold_CV(0.5))
	print('Threshold CV 0.4', evaluate_threshold_CV(0.4))

	#calculate the area under the curve to get the performance for a classifier
	# IMPORTANT: first argument is true values, second argument is predicted probabilities
	print('AUC for test set', metrics.roc_auc_score(y_test, y_pred_prob))
	print('AUC for CV train set', metrics.roc_auc_score(y_train, y_scores[:,1]))

	# calculate cross_val_scores with different scoring functions for test set
	from sklearn.model_selection import cross_val_score
	train_roc_auc = cross_val_score(tree_clf_new, X_transform_test, y_test,
									scoring='roc_auc').mean()
	train_accuracy = cross_val_score(tree_clf_new, X_transform_test, y_test,
									scoring='accuracy').mean()
	train_recall = cross_val_score(tree_clf_new, X_transform_test, y_test,
									scoring='recall').mean()
	train_precision = cross_val_score(tree_clf_new, X_transform_test, y_test,
									scoring='precision').mean()
	train_f1 = cross_val_score(tree_clf_new, X_transform_test, y_test,
									scoring='f1').mean()

	print('ROC_AUC', train_roc_auc)#uses metrics.roc_auc_score
	print('Accuracy', train_accuracy)#uses metrics.accuracy_score
	print('Recall', train_recall)#uses metrics.recall_score
	print('Precision', train_precision)#uses metrics.precision_score
	print('F1 score', train_f1)#uses metrics.f1_score

	# calculate cross-validated AUC for CV train set
	from sklearn.model_selection import cross_val_score
	train_roc_auc = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='roc_auc').mean()
	train_accuracy = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='accuracy').mean()
	train_recall = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='recall').mean()
	train_precision = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='precision').mean()
	train_f1 = cross_val_score(tree_clf_new, X_transform_train, y_train, cv=10,
									scoring='f1').mean()

	print('ROC_AUC CV', train_roc_auc)#uses metrics.roc_auc_score
	print('Accuracy CV', train_accuracy)#uses metrics.accuracy_score
	print('Recall CV', train_recall)#uses metrics.recall_score
	print('Precision CV', train_precision)#uses metrics.precision_score
	print('F1 score CV', train_f1)#uses metrics.f1_score

	# calculate cross-validated AUC for CV TEST set
	from sklearn.model_selection import cross_val_score
	train_roc_auc = cross_val_score(tree_clf_new, X_transform_test, y_test, cv=10,
									scoring='roc_auc').mean()
	train_accuracy = cross_val_score(tree_clf_new, X_transform_test, y_test, cv=10,
									scoring='accuracy').mean()
	train_recall = cross_val_score(tree_clf_new, X_transform_test, y_test, cv=10,
									scoring='recall').mean()
	train_precision = cross_val_score(tree_clf_new, X_transform_test, y_test, cv=10,
									scoring='precision').mean()
	train_f1 = cross_val_score(tree_clf_new, X_transform_test, y_test, cv=10,
									scoring='f1').mean()

	print('ROC_AUC CV TEST', train_roc_auc)#uses metrics.roc_auc_score
	print('Accuracy CV TEST', train_accuracy)#uses metrics.accuracy_score
	print('Recall CV TEST', train_recall)#uses metrics.recall_score
	print('Precision CV TEST', train_precision)#uses metrics.precision_score
	print('F1 score CV TEST', train_f1)#uses metrics.f1_score

















