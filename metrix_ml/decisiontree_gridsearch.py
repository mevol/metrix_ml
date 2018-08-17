import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.tree import export_graphviz



def parse_command_line():
	
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

def load_metrix_data(csv_path):
		return pd.read_csv(csv_path)


class DecisionTreeGridSearch(object):
	def __init__(self, metrix, outdir):
		self.metrix=metrix
		self.outdir=outdir
		self.prepare_metrix_data()
		self.split_data()
		self.grid_search()
		self.tree_best_params()
		self.predict()
		self.analysis()


	def prepare_metrix_data(self):
			###############################################################################
		#
		#  creating 3 data frames specific to the three development milestones I had
		#  1--> directly from data processing
		#  2--> after adding protein information
		#  3--> carrying out some further column transformations
		#
      
		#look at the data that is coming from processing
		attr_database = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
										 'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
										 'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
										 'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF']
		metrix_database = self.metrix[attr_database]
		
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Preparing input data as metrix_database with following attributes %s \n' %(attr_database))


		#database plus manually added data
		attr_man_add = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
										'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
										'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
										'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF',
										'wavelength', 'Vcell', 'Matth_coeff', 'No_atom_chain', 'solvent_content',
										'No_mol_ASU', 'MW_chain', 'sites_ASU']
		metrix_man_add = self.metrix[attr_man_add]

		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
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

		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
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

		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Created the following dataframes: metrix_database, metrix_man_add, metrix_transform \n')


	def split_data(self):
			###############################################################################
		#
		#  creating training and test set for each of the 3 dataframes
		#
		###############################################################################

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

		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Spliting into training and test set 80-20 \n')
		  text_file.write('metrix_database: X_database_train, X_database_test \n')
		  text_file.write('metrix_man_add: X_man_add_train, X_man_add_test \n')
		  text_file.write('metrix_transform: X_transform_train, X_transform_test \n')
		  text_file.write('y(EP_success): y_train, y_test \n')


	def grid_search(self):
			#training a decision tree with the prepared train set and train set lables


		#create the decision tree
		tree_clf = DecisionTreeClassifier(random_state=42)

		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Created decision tree: tree_clf \n')

		#set up grid search
		param_grid = {"criterion": ["gini", "entropy"],
									'max_features': [1, 2, 4, 8, 16],
									 "min_samples_split": [5, 10, 15], #min samples per node to induce split
									 "max_depth": [3, 4, 5, 6], #max number of splits to do
									 "min_samples_leaf": [2, 4, 6], #min number of samples in a leaf
									 "max_leaf_nodes": [5, 10, 15]}#max number of leaves

		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Running grid search for the following parameters: %s \n' %param_grid)
		  text_file.write('use cv=10, scoring=accuracy \n')
						 

		#building and running the grid search
		grid_search = GridSearchCV(tree_clf, param_grid, cv=10,
															scoring='accuracy')

		grid_search.fit(self.X_transform_train, self.y_train)

		#get best parameter combination and its score as accuracy
		print(grid_search.best_params_)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Best parameters: ' +str(grid_search.best_params_)+'\n')
		
		print(grid_search.best_score_)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Best score: ' +str(grid_search.best_score_)+'\n')
		
		feature_importances = grid_search.best_estimator_.feature_importances_
		feature_importances_ls = sorted(zip(feature_importances, self.X_transform_train), reverse=True)
		print(sorted(zip(feature_importances, self.X_transform_train), reverse=True))
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Feature importances: %s \n' %feature_importances_ls)

		
		self.best_params = grid_search.best_params_
	
	
	
	def tree_best_params(self):
		self.tree_clf_new = DecisionTreeClassifier(**self.best_params, random_state=42)

		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Created new decision tree "tree_clf_new" using best parameters \n')

		#visualise best decision tree

		self.tree_clf_new.fit(self.X_transform_train, self.y_train)
		dotfile = os.path.join(self.outdir, 'tree_clf_new.dot')
		pngfile = os.path.join(self.outdir, 'tree_clf_new.png')

		with open(dotfile, 'w') as f:
				export_graphviz(self.tree_clf_new, out_file=f, feature_names=self.X_transform_train.columns,
											 rounded=True, filled=True)
												
		command = ["dot", "-Tpng", dotfile, "-o", pngfile]
		subprocess.check_call(command)

		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Writing DOTfile and convert to PNG for "tree_clf_new" \n')
		  text_file.write('DOT filename: tree_clf_new.dot \n')
		  text_file.write('PNG filename: tree_clf_new.png \n')



		#not the best measure to use as it heavily depends on the sample 
		#distribution --> accuracy
		print(cross_val_score(self.tree_clf_new, self.X_transform_train, self.y_train,
										cv=10, scoring='accuracy'))
		accuracy_each_cv = cross_val_score(self.tree_clf_new, self.X_transform_train, self.y_train,
										cv=10, scoring='accuracy')
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Accuracy for each of 10 CV folds: %s \n' %accuracy_each_cv)
										
		print(cross_val_score(self.tree_clf_new, self.X_transform_train, self.y_train,
										cv=10, scoring='accuracy').mean())
		accuracy_mean_cv = cross_val_score(self.tree_clf_new, self.X_transform_train, self.y_train,
										cv=10, scoring='accuracy').mean()
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Mean accuracy over all 10 CV folds: %s \n' %accuracy_mean_cv)


		# calculate cross_val_scoring with different scoring functions for CV train set
		train_roc_auc = cross_val_score(self.tree_clf_new, self.X_transform_train, self.y_train, cv=10,
										scoring='roc_auc').mean()
		print('ROC_AUC CV', train_roc_auc)#uses metrics.roc_auc_score
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('ROC_AUC mean for 10-fold CV: %s \n' %train_roc_auc)
										
		train_accuracy = cross_val_score(self.tree_clf_new, self.X_transform_train, self.y_train, cv=10,
										scoring='accuracy').mean()
		print('Accuracy CV', train_accuracy)#uses metrics.accuracy_score
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Accuracy mean for 10-fold CV: %s \n' %train_accuracy)
										
		train_recall = cross_val_score(self.tree_clf_new, self.X_transform_train, self.y_train, cv=10,
										scoring='recall').mean()
		print('Recall CV', train_recall)#uses metrics.recall_score
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Recall mean for 10-fold CV: %s \n' %train_recall)

		train_precision = cross_val_score(self.tree_clf_new, self.X_transform_train, self.y_train, cv=10,
										scoring='precision').mean()
		print('Precision CV', train_precision)#uses metrics.precision_score
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Precision mean for 10-fold CV: %s \n' %train_precision)

		train_f1 = cross_val_score(self.tree_clf_new, self.X_transform_train, self.y_train, cv=10,
										scoring='f1').mean()
		print('F1 score CV', train_f1)#uses metrics.f1_score
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('F1 score mean for 10-fold CV: %s \n' %train_f1)


	def predict(self):
		#try out how well the classifier works to predict from the test set
		self.y_pred_class = self.tree_clf_new.predict(self.X_transform_test)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Saving predictions for X_transform_test in y_pred_class \n')

		#alternative way to not have to use the test set
		self.y_train_pred = cross_val_predict(self.tree_clf_new, self.X_transform_train, self.y_train,
											cv=10)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Saving predictions for X_transform_train with 10-fold CV in y_train_pred \n')

		# calculate accuracy
		y_accuracy = metrics.accuracy_score(self.y_test, self.y_pred_class)
		print(metrics.accuracy_score(self.y_test, self.y_pred_class))
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Accuracy score or agreement between y_test and y_pred_class: %s \n' %y_accuracy)

		# examine the class distribution of the testing set (using a Pandas Series method)
		class_dist = self.y_test.value_counts()
		print(self.y_test.value_counts())
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Class distribution for y_test: %s \n' %class_dist)

		# calculate the percentage of ones
		# because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
		ones = self.y_test.mean()
		print(self.y_test.mean())
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Percent 1s in y_test: %s \n' %ones)

		# calculate the percentage of zeros
		zeros = 1 - self.y_test.mean()
		print(1 - self.y_test.mean())
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Percent 0s in y_test: %s \n' %zeros)

		# calculate null accuracy in a single line of code
		# only for binary classification problems coded as 0/1
		null_acc = max(self.y_test.mean(), 1 - self.y_test.mean())
		print(null_acc)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Null accuracy in y_test: %s \n' %null_acc)

	def analysis(self):
		# IMPORTANT: first argument is true values, second argument is predicted values
		# this produces a 2x2 numpy array (matrix)
		conf_mat_test = metrics.confusion_matrix(self.y_test, self.y_pred_class)
		conf_mat_10CV = metrics.confusion_matrix(self.y_train, self.y_train_pred)
		print('confusion matrix using test set %s' %conf_mat_test)#on the test set
		print('confusion matrix using CV train set %s' %conf_mat_10CV)#on the CV train set
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('confusion matrix using test set: %s \n' %conf_mat_test)
		  text_file.write('confusion matrix using 10-fold CV: %s \n' %conf_mat_10CV)


		# slice confusion matrix into four pieces
		#[row, column] for test set
		TP = conf_mat_test[1, 1]
		TN = conf_mat_test[0, 0]
		FP = conf_mat_test[0, 1]
		FN = conf_mat_test[1, 0]
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Slicing confusion matrix for test set into: TP, TN, FP, FN \n')

		#[row, column] for CV train set
		TP_CV = conf_mat_10CV[1, 1]
		TN_CV = conf_mat_10CV[0, 0]
		FP_CV = conf_mat_10CV[0, 1]
		FN_CV = conf_mat_10CV[1, 0]
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Slicing confusion matrix for 10-fold CV into: TP_CV, TN_CV, FP_CV, FN_CV \n')

		#metrics calculated from confusion matrix
		# use float to perform true division, not integer division
		acc_score_man_test = (TP + TN) / float(TP + TN + FP + FN)
		acc_score_sklearn_test = metrics.accuracy_score(self.y_test, self.y_pred_class)
		acc_score_man_CV = (TP_CV + TN_CV) / float(TP_CV + TN_CV + FP_CV + FN_CV)
		acc_score_sklearn_CV = metrics.accuracy_score(self.y_train, self.y_train_pred)
		print('accuracy score manual test: %s' %acc_score_man_test)
		print('accuracy score sklearn test: %s' %acc_score_sklearn_test)
		print('accuracy score manual CV: %s' %acc_score_man_CV)
		print('accuracy score sklearn CV: %s' %acc_score_sklearn_CV)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
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
		print('classification error manual test: %s' %class_err_man_test)
		print('classification error sklearn test: %s' %class_err_sklearn_test)
		print('classification error manual CV: %s' %class_err_man_CV)
		print('classification error sklearn CV: %s' %class_err_sklearn_CV)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
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
		print('sensitivity manual test: %s' %sensitivity_man_test)
		print('sensitivity sklearn test: %s' %sensitivity_sklearn_test)
		print('sensitivity manual CV: %s' %sensitivity_man_CV)
		print('sensitivity sklearn CV: %s' %sensitivity_sklearn_CV)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Sensitivity/Recall/True positives: \n')
		  text_file.write('sensitivity manual test: %s \n' %sensitivity_man_test)
		  text_file.write('sensitivity sklearn test: %s \n' %sensitivity_sklearn_test)
		  text_file.write('sensitivity manual CV: %s \n' %sensitivity_man_CV)
		  text_file.write('sensitivity sklearn CV: %s \n' %sensitivity_sklearn_CV)
  
    #calculate specificity
		specificity_man_test = TN / (TN + FP)
		specificity_man_CV = TN_CV / (TN_CV + FP_CV)
		print('specificity manual test: %s' %specificity_man_test)
		print('specificity manual CV: %s' %specificity_man_CV)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Specificity: \n')
		  text_file.write('specificity manual test: %s \n' %specificity_man_test)
		  text_file.write('specificity manual CV: %s \n' %specificity_man_CV)
    
    #calculate false positive rate
		false_positive_rate_man_test = FP / float(TN + FP)
		false_positive_rate_man_CV = FP_CV / float(TN_CV + FP_CV)
		print('false positive rate manual test: %s' %false_positive_rate_man_test)
		print('1 - specificity test: %s' %(1 - specificity_man_test))
		print('false positive rate manual CV: %s' %false_positive_rate_man_CV)
		print('1 - specificity CV: %s' %(1 - specificity_man_CV))
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
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
		print('precision manual: %s' %precision_man_test)
		print('precision sklearn: %s' %precision_sklearn_test)
		print('precision manual CV: %s' %precision_man_CV)
		print('precision sklearn CV: %s' %precision_sklearn_CV)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Precision or confidence of classification: \n')
		  text_file.write('precision manual: %s \n' %precision_man_test)
		  text_file.write('precision sklearn: %s \n' %precision_sklearn_test)
		  text_file.write('precision manual CV: %s \n' %precision_man_CV)
		  text_file.write('precision sklearn CV: %s \n' %precision_sklearn_CV)

		#F1 score; uses precision and recall
		f1_score_sklearn_test = f1_score(self.y_test, self.y_pred_class)
		f1_score_sklearn_CV = f1_score(self.y_train, self.y_train_pred)
		print('F1 score sklearn test: %s' %f1_score_sklearn_test)
		print('F1 score sklearn CV: %s' %f1_score_sklearn_CV)
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('F1 score: \n')
		  text_file.write('F1 score sklearn test: %s \n' %f1_score_sklearn_test)
		  text_file.write('F1 score sklearn CV: %s \n' %f1_score_sklearn_CV)

		###############################################################################
		#
		#  adjusting classification thresholds; default threshold is 0.5
		#
		###############################################################################


		#probabilities for the CV train set
		self.y_probas_CV = cross_val_predict(self.tree_clf_new, self.X_transform_train, self.y_train, cv=10, method='predict_proba')

#		self.y_scores=self.tree_clf_new.predict_proba(self.X_transform_train)#train set
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Storing prediction probabilities for X_transform_train and y_train with 10-fold CV in y_probas_CV \n')

		# store the predicted probabilities for class 1
		self.y_scores_CV = self.y_probas_CV[:, 1]
#		self.y_pred_prob = self.tree_clf_new.predict_proba(self.X_transform_test)[:, 1]#test set
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Storing prediction probabilities for class 1 for X_transform_train and y_train in y_scores_CV \n')
		# histogram of predicted probabilities

		# 8 bins for prediction probability on the test set
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Plotting histogram for y_pred_proba\n')
		plt.hist(self.y_probas_CV, bins=8)
#		plt.hist(self.y_pred_prob, bins=8)
		# x-axis limit from 0 to 1
		plt.xlim(0,1)
		plt.title('Histogram of predicted probabilities for y_pred_prob or class 1')
		plt.xlabel('Predicted probability of EP_success')
		plt.ylabel('Frequency')
		plt.show()

		# 8 bins for prediction probability on the CV train set
		with open(os.path.join(self.outdir, 'decisiontree_gridsearch.txt'), 'a') as text_file:
		  text_file.write('Plotting histogram for y_scores\n')
		plt.hist(self.y_scores_CV, bins=8)
#		plt.hist(self.y_scores[:,1], bins=8)
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
		precisions, recalls, thresholds_forest = precision_recall_curve(self.y_test, self.y_pred_prob)

		def plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest):
				plt.plot(thresholds_forest, precisions[:-1], "b--", label="Precision")
				plt.plot(thresholds_forest, recalls[:-1], "g--", label="Recall")
				plt.xlabel("Threshold")
				plt.legend(loc="upper left")
				plt.ylim([0,1])

		plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest)
		plt.show()

		#plot Precision Recall Threshold curve for CV train set 
		precisions, recalls, thresholds_forest = precision_recall_curve(self.y_train, self.y_scores[:,1])

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
		fpr, tpr, thresholds = metrics.roc_curve(self.y_test, self.y_pred_prob)#test set

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
		fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_prob)#test set

		def plot_roc_curve(fpr, tpr, label=None):
				plt.plot(fpr, tpr, linewidth=2, label=label)
				plt.plot([0, 1], [0, 1], 'k--')
				plt.axis([0, 1, 0, 1])
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
		
		plot_roc_curve(fpr, tpr)
		plt.show()

		#for the CV training set
		fpr_CV, tpr_CV, thresholds_CV = metrics.roc_curve(self.y_train, self.y_scores[:,1])#CV train set

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
		fpr_CV, tpr_CV, thresholds_CV = roc_curve(self.y_train, self.y_scores[:,1])#CV train set

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
		print('AUC for test set', metrics.roc_auc_score(self.y_test, self.y_pred_prob))
		print('AUC for CV train set', metrics.roc_auc_score(self.y_train, self.y_scores[:,1]))
		
		
		def scoring(X, y, cv):
			# calculate cross_val_scores with different scoring functions for test set
			roc_auc = cross_val_score(self.tree_clf_new, X, y, cv=cv,
											scoring='roc_auc').mean()
			accuracy = cross_val_score(self.tree_clf_new, X, y, cv=cv,
											scoring='accuracy').mean()
			recall = cross_val_score(self.tree_clf_new, X, y, cv=cv,
											scoring='recall').mean()
			precision = cross_val_score(self.tree_clf_new, X, y, cv=cv,
											scoring='precision').mean()
			f1 = cross_val_score(self.tree_clf_new, X, y, cv=cv,
											scoring='f1').mean()

			print('ROC_AUC', roc_auc)#uses metrics.roc_auc_score
			print('Accuracy', accuracy)#uses metrics.accuracy_score
			print('Recall', recall)#uses metrics.recall_score
			print('Precision', precision)#uses metrics.precision_score
			print('F1 score', f1)#uses metrics.f1_score

		scoring(self.X_transform_test, self.y_test, cv=None)
		scoring(self.X_transform_train, self.y_train, cv=10)

def run():
	args = parse_command_line()
	
	
	###############################################################################

	#look at the imported data to get an idea what we are working with
	metrix = load_metrix_data(args.input)

	###############################################################################

	decision_tree_grid_search = DecisionTreeGridSearch(metrix, args.outdir)



















