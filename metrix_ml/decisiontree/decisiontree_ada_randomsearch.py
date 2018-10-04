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
from sklearn.ensemble import AdaBoostClassifier

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
#  load the data from CSV file
#
###############################################################################

def load_metrix_data(csv_path):
  '''load the raw data as stored in CSV file'''
  return pd.read_csv(csv_path)
  
def make_output_folder(outdir):
  names = ['database', 'man_add', 'transform', 'prot_screen_trans']
  result = []
  for name in names:
    name = os.path.join(outdir, 'randomforest_ada_randomsearch', name)
    os.makedirs(name, exist_ok=True)
    result.append(name)
  return result

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
  def __init__(self, metrix, database, man_add, transform, prot_screen_trans):
    self.metrix=metrix
    self.database=database
    self.man_add=man_add
    self.transform=transform
    self.prot_screen_trans=prot_screen_trans
    self.prepare_metrix_data()
    self.split_data()
    self.rand_search()
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
    print('*    Preparing input dataframes metrix_database, metrix_man_add, metrix_transform, metrix_prot_screen_trans')
    print('*' *80)

    #look at the data that is coming from processing
    attr_database = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
                      'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
                      'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
                      'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF']
    metrix_database = self.metrix[attr_database]
    
    with open(os.path.join(self.database, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_database with following attributes %s \n' %(attr_database))

    #database plus manually added data
    attr_man_add = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
                    'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
                    'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
                    'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF',
                    'wavelength', 'Vcell', 'Matth_coeff', 'No_atom_chain', 'solvent_content',
                    'No_mol_ASU', 'MW_chain', 'sites_ASU']
    metrix_man_add = self.metrix[attr_man_add]

    with open(os.path.join(self.man_add, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
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
                      'MW_chain/No_atom_chain', 'wilson', 'bragg',
                      'volume_wilsonB_highres']                          

    attr_prot_screen_trans = ['highreslimit', 'wavelength', 'Vcell', 'wavelength**3',
                         'wavelength**3/Vcell', 'solvent_content', 'Vcell/Vm<Ma>',
                         'Matth_coeff', 'MW_ASU/sites_ASU/solvent_content',
                         'MW_chain', 'No_atom_chain', 'No_mol_ASU', 'MW_ASU',
                         'sites_ASU', 'MW_ASU/sites_ASU', 'MW_chain/No_atom_chain']

    metrix_transform = metrix_man_add.copy()
    metrix_prot_screen_trans = metrix_man_add[['highreslimit', 'wavelength', 'Vcell',
    'Matth_coeff', 'No_atom_chain', 'solvent_content', 'No_mol_ASU', 'MW_chain', 'sites_ASU']].copy()

    with open(os.path.join(self.transform, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_transform with following attributes %s \n' %(attr_transform))

    with open(os.path.join(self.prot_screen_trans, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_transform with following attributes %s \n' %(attr_prot_screen_trans))

    #column transformation
    #MW_ASU
    metrix_transform['MW_ASU'] = metrix_transform['MW_chain'] * metrix_transform['No_mol_ASU']
    metrix_prot_screen_trans['MW_ASU'] = metrix_prot_screen_trans['MW_chain'] * metrix_prot_screen_trans['No_mol_ASU']

    #MW_ASU/sites_ASU
    metrix_transform['MW_ASU/sites_ASU'] = metrix_transform['MW_ASU'] / metrix_transform['sites_ASU']
    metrix_prot_screen_trans['MW_ASU/sites_ASU'] = metrix_prot_screen_trans['MW_ASU'] / metrix_prot_screen_trans['sites_ASU']

    #MW_chain/No_atom_chain
    metrix_transform['MW_chain/No_atom_chain'] = metrix_transform['MW_chain'] / metrix_transform['No_atom_chain']
    metrix_prot_screen_trans['MW_chain/No_atom_chain'] = metrix_prot_screen_trans['MW_chain'] / metrix_prot_screen_trans['No_atom_chain']

    #MW_ASU/sites_ASU/solvent_content
    metrix_transform['MW_ASU/sites_ASU/solvent_content'] = metrix_transform['MW_ASU/sites_ASU'] / metrix_transform['solvent_content']
    metrix_prot_screen_trans['MW_ASU/sites_ASU/solvent_content'] = metrix_prot_screen_trans['MW_ASU/sites_ASU'] / metrix_prot_screen_trans['solvent_content']

    #wavelength**3
    metrix_transform['wavelength**3'] = metrix_transform['wavelength'] ** 3
    metrix_prot_screen_trans['wavelength**3'] = metrix_prot_screen_trans['wavelength'] ** 3

    #wavelenght**3/Vcell
    metrix_transform['wavelength**3/Vcell'] = metrix_transform['wavelength**3'] / metrix_transform['Vcell']
    metrix_prot_screen_trans['wavelength**3/Vcell'] = metrix_prot_screen_trans['wavelength**3'] / metrix_prot_screen_trans['Vcell']

    #Vcell/Vm<Ma>
    metrix_transform['Vcell/Vm<Ma>'] = metrix_transform['Vcell'] / (metrix_transform['Matth_coeff'] * metrix_transform['MW_chain/No_atom_chain'])
    metrix_prot_screen_trans['Vcell/Vm<Ma>'] = metrix_prot_screen_trans['Vcell'] / (metrix_prot_screen_trans['Matth_coeff'] * metrix_prot_screen_trans['MW_chain/No_atom_chain'])

    #wilson
    metrix_transform['wilson'] = -2 * metrix_transform['wilsonbfactor']

    #bragg
    metrix_transform['bragg'] = (1 / metrix_transform['highreslimit'])**2

    #use np.exp to work with series object
    metrix_transform['volume_wilsonB_highres'] = metrix_transform['Vcell/Vm<Ma>'] * np.exp(metrix_transform['wilson'] * metrix_transform['bragg'])
    
    self.X_database = metrix_database
    self.X_man_add = metrix_man_add
    self.X_transform = metrix_transform
    self.X_prot_screen_trans = metrix_prot_screen_trans

    with open(os.path.join(self.database, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_database \n')
    with open(os.path.join(self.man_add, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_man_add \n')
    with open(os.path.join(self.transform, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_transform \n')
    with open(os.path.join(self.prot_screen_trans, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_prot_screen_trans \n')

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
    X_prot_screen_trans_train, X_prot_screen_trans_test, y_train, y_test = train_test_split(self.X_prot_screen_trans, y, test_size=0.2, random_state=42)

    assert self.X_database.columns.all() == X_database_train.columns.all()
    assert self.X_man_add.columns.all() == X_man_add_train.columns.all()
    assert self.X_transform.columns.all() == X_transform_train.columns.all()
    self.X_database_train = X_database_train
    self.X_man_add_train = X_man_add_train
    self.X_transform_train = X_transform_train
    self.X_prot_screen_trans_train = X_prot_screen_trans_train    
    self.X_database_test = X_database_test
    self.X_man_add_test = X_man_add_test
    self.X_transform_test = X_transform_test
    self.X_prot_screen_trans_test = X_prot_screen_trans_test    
    self.y_train = y_train
    self.y_test = y_test

    with open(os.path.join(self.database, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_database: X_database_train, X_database_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')
      
    with open(os.path.join(self.man_add, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_man_add: X_man_add_train, X_man_add_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')
      
    with open(os.path.join(self.transform, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_transform: X_transform_train, X_transform_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')

    with open(os.path.join(self.prot_screen_trans, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('metrix_transform: X_prot_screen_trans_train, X_prot_screen_trans_test \n')
      text_file.write('y(EP_success): y_train, y_test \n')

    ###############################################################################
    #
    #  random search for best parameter combination
    #
    ###############################################################################

  def rand_search(self):
    '''running a randomized search to find the parameter combination for a random forest
     which gives the best accuracy score'''
    print('*' *80)
    print('*    Running RandomizedSearch for best parameter combination for RandomForest')
    print('*' *80)

    #create the decision forest
    #forest_clf_rand_ada = RandomForestClassifier(random_state=42)
    tree_clf_rand_ada = DecisionTreeClassifier(random_state=42)
    
    with open(os.path.join(self.database, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created random forest: forest_clf_rand_ada \n')
    with open(os.path.join(self.man_add, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created random forest: forest_clf_rand_ada \n')
    with open(os.path.join(self.transform, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created random forest: forest_clf_rand_ada \n')
    with open(os.path.join(self.prot_screen_trans, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Created random forest: forest_clf_rand_ada \n')

    #set up randomized search
    param_rand = {"criterion": ["gini", "entropy"],
                  #'n_estimators': randint(10, 500),
                  'max_features': randint(2, 16),
                  "min_samples_split": randint(2, 20),
                  "max_depth": randint(5, 10),
                  "min_samples_leaf": randint(1, 20),
                  "max_leaf_nodes": randint(10, 20)}

    with open(os.path.join(self.database, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Running randomised search for the following parameters: %s \n' %param_rand)
      text_file.write('use cv=10, scoring=accuracy \n')
    with open(os.path.join(self.man_add, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Running randomised search for the following parameters: %s \n' %param_rand)
      text_file.write('use cv=10, scoring=accuracy \n')
    with open(os.path.join(self.transform, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Running randomised search for the following parameters: %s \n' %param_rand)
      text_file.write('use cv=10, scoring=accuracy \n')
    with open(os.path.join(self.prot_screen_trans, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Running randomised search for the following parameters: %s \n' %param_rand)
      text_file.write('use cv=10, scoring=accuracy \n')

    #building and running the randomized search
    rand_search = RandomizedSearchCV(tree_clf_rand_ada, param_rand, random_state=5,
                              cv=10, n_iter=288, scoring='accuracy')

    rand_search_database = rand_search.fit(self.X_database_train, self.y_train)
    with open(os.path.join(self.database, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Best parameters: ' +str(rand_search_database.best_params_)+'\n')
      text_file.write('Best score: ' +str(rand_search_database.best_score_)+'\n')
    feature_importances_database = rand_search_database.best_estimator_.feature_importances_
    feature_importances_database_ls = sorted(zip(feature_importances_database, self.X_database_train), reverse=True)
    with open(os.path.join(self.database, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Feature importances: %s \n' %feature_importances_database_ls)

    rand_search_man_add = rand_search.fit(self.X_man_add_train, self.y_train)
    with open(os.path.join(self.man_add, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Best parameters: ' +str(rand_search_man_add.best_params_)+'\n')
      text_file.write('Best score: ' +str(rand_search_man_add.best_score_)+'\n')
    feature_importances_man_add = rand_search_man_add.best_estimator_.feature_importances_
    feature_importances_man_add_ls = sorted(zip(feature_importances_man_add, self.X_man_add_train), reverse=True)
    with open(os.path.join(self.man_add, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Feature importances: %s \n' %feature_importances_man_add_ls)

    rand_search_transform = rand_search.fit(self.X_transform_train, self.y_train)
    with open(os.path.join(self.transform, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Best parameters: ' +str(rand_search_transform.best_params_)+'\n')
      text_file.write('Best score: ' +str(rand_search_transform.best_score_)+'\n')
    feature_importances_transform = rand_search_transform.best_estimator_.feature_importances_
    feature_importances_transform_ls = sorted(zip(feature_importances_transform, self.X_transform_train), reverse=True)
    with open(os.path.join(self.transform, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Feature importances: %s \n' %feature_importances_transform_ls)

    rand_search_prot_screen_trans = rand_search.fit(self.X_prot_screen_trans_train, self.y_train)
    with open(os.path.join(self.prot_screen_trans, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Best parameters: ' +str(rand_search_prot_screen_trans.best_params_)+'\n')
      text_file.write('Best score: ' +str(rand_search_prot_screen_trans.best_score_)+'\n')
    feature_importances_prot_screen_trans = rand_search_prot_screen_trans.best_estimator_.feature_importances_
    feature_importances_prot_screen_trans_ls = sorted(zip(feature_importances_prot_screen_trans, self.X_prot_screen_trans_train), reverse=True)
    with open(os.path.join(self.prot_screen_trans, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Feature importances: %s \n' %feature_importances_prot_screen_trans_ls)
    
    self.best_params_database = rand_search_database.best_params_
    self.best_params_man_add = rand_search_man_add.best_params_
    self.best_params_transform = rand_search_transform.best_params_
    self.best_params_prot_screen_trans = rand_search_prot_screen_trans.best_params_

    ###############################################################################
    #
    #  creating new forest with best parameter combination
    #
    ###############################################################################

  def forest_best_params(self):
    '''create a new random forest using the best parameter combination found above'''
    print('*' *80)
    print('*    Building new forest based on best parameter combination and save as pickle')
    print('*' *80)
    #
    self.tree_clf_rand_ada_new_database = AdaBoostClassifier(
                                DecisionTreeClassifier(**self.best_params_database, random_state=42),
                                algorithm ="SAMME.R", learning_rate=0.5, random_state=5)
    self.tree_clf_rand_ada_new_database.fit(self.X_database_train, self.y_train)

    self.tree_clf_rand_ada_new_man_add = AdaBoostClassifier(
                                DecisionTreeClassifier(**self.best_params_man_add, random_state=42),
                                algorithm ="SAMME.R", learning_rate=0.5, random_state=5)
    self.tree_clf_rand_ada_new_man_add.fit(self.X_man_add_train, self.y_train)

    self.tree_clf_rand_ada_new_transform = AdaBoostClassifier(
                                DecisionTreeClassifier(**self.best_params_transform, random_state=42),
                                algorithm ="SAMME.R", learning_rate=0.5, random_state=5)
    self.tree_clf_rand_ada_new_transform.fit(self.X_transform_train, self.y_train)

    self.tree_clf_rand_ada_new_prot_screen_trans = AdaBoostClassifier(
                                DecisionTreeClassifier(**self.best_params_prot_screen_trans, random_state=42),
                                algorithm ="SAMME.R", learning_rate=0.5, random_state=5)
    self.tree_clf_rand_ada_new_prot_screen_trans.fit(self.X_prot_screen_trans_train, self.y_train)

    def feature_importances(clf, X_train, directory):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      importances = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)     
      indices = np.argsort(importances)
      feature_names = X_train.columns
      std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)#for random forest
     # Plot the feature importances of the forest
      plt.figure(figsize=(20,10))
      plt.title("Feature importances")
      plt.bar(range(X_train.shape[1]), importances[indices],
         color="b", yerr=std[indices], align="center")#add yerr=std[indices] for random forest
      plt.xticks(range(X_train.shape[1]), feature_names,rotation=60)
      plt.xlim([-1, X_train.shape[1]])
      plt.savefig(os.path.join(directory, 'feature_importances_bar_plot_rand_ada_'+datestring+'.png'))
      plt.close()

    feature_importances(self.tree_clf_rand_ada_new_database, self.X_database_train, self.database)
    feature_importances(self.tree_clf_rand_ada_new_man_add, self.X_man_add_train, self.man_add)
    feature_importances(self.tree_clf_rand_ada_new_transform, self.X_transform_train, self.transform)
    feature_importances(self.tree_clf_rand_ada_new_prot_screen_trans, self.X_prot_screen_trans_train, self.prot_screen_trans)

    def write_pickle(forest, directory, name):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      joblib.dump(forest, os.path.join(directory,'best_forest_rand_ada_'+name+datestring+'.pkl'))
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Created new random forest "forest_clf_rand_ada_new_%s" using best parameters \n' %name)
        text_file.write('Creating pickle file for best forest as best_forest_rand_ada_%s.pkl \n' %name)
    
    write_pickle(self.tree_clf_rand_ada_new_database, self.database, 'database')
    write_pickle(self.tree_clf_rand_ada_new_man_add, self.man_add, 'man_add')
    write_pickle(self.tree_clf_rand_ada_new_transform, self.transform, 'transform')
    write_pickle(self.tree_clf_rand_ada_new_prot_screen_trans, self.prot_screen_trans, 'prot_screen_trans')

    def visualise_tree(tree_forest, directory, columns, name):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      trees = tree_forest.estimators_
      i_tree = 0
      for tree in trees:
        with open(os.path.join(directory,'tree_clf_rand_ada_new_'+name+datestring+str(i_tree)+'.dot'), 'w') as f:
          export_graphviz(tree, out_file=f, feature_names=columns, rounded=True, filled=True)
          f.close()
        dotfile = os.path.join(directory, 'tree_clf_rand_ada_new_'+name+datestring+str(i_tree)+'.dot')
        pngfile = os.path.join(directory, 'tree_clf_rand_ada_new_'+name+datestring+str(i_tree)+'.png')
        command = ["dot", "-Tpng", dotfile, "-o", pngfile]
        subprocess.check_call(command)
        i_tree = i_tree + 1

      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Writing DOTfile and convert to PNG for "tree_clf_rand_ada_new_%s" \n' %name)
        text_file.write('DOT filename: tree_clf_rand_ada_new_%s.dot \n' %name)
        text_file.write('PNG filename: tree_clf_rand_ada_new_%s.png \n' %name)

    
    visualise_tree(self.tree_clf_rand_ada_new_database, self.database, self.X_database_train.columns, 'database')
    visualise_tree(self.tree_clf_rand_ada_new_man_add, self.man_add, self.X_man_add_train.columns, 'man_add')
    visualise_tree(self.tree_clf_rand_ada_new_transform, self.transform, self.X_transform_train.columns, 'transform')
    visualise_tree(self.tree_clf_rand_ada_new_prot_screen_trans, self.prot_screen_trans, self.X_prot_screen_trans_train.columns, 'prot_screen_trans')

    print('*' *80)
    print('*    Getting basic stats for new forest')
    print('*' *80)

    def basic_stats(forest, X_train, directory):
      #distribution --> accuracy
      accuracy_each_cv = cross_val_score(forest, X_train, self.y_train, cv=10, scoring='accuracy')
      accuracy_mean_cv = cross_val_score(forest, X_train, self.y_train, cv=10, scoring='accuracy').mean()
      # calculate cross_val_scoring with different scoring functions for CV train set
      train_roc_auc = cross_val_score(forest, X_train, self.y_train, cv=10, scoring='roc_auc').mean()
      train_accuracy = cross_val_score(forest, X_train, self.y_train, cv=10, scoring='accuracy').mean()
      train_recall = cross_val_score(forest, X_train, self.y_train, cv=10, scoring='recall').mean()
      train_precision = cross_val_score(forest, X_train, self.y_train, cv=10, scoring='precision').mean()
      train_f1 = cross_val_score(forest, X_train, self.y_train, cv=10, scoring='f1').mean()

      with open(os.path.join(directory, 'randomforest_randomsearch.txt'), 'a') as text_file:
        text_file.write('Accuracy for each of 10 CV folds: %s \n' %accuracy_each_cv)
        text_file.write('Mean accuracy over all 10 CV folds: %s \n' %accuracy_mean_cv)
        text_file.write('ROC_AUC mean for 10-fold CV: %s \n' %train_roc_auc)
        text_file.write('Accuracy mean for 10-fold CV: %s \n' %train_accuracy)
        text_file.write('Recall mean for 10-fold CV: %s \n' %train_recall)
        text_file.write('Precision mean for 10-fold CV: %s \n' %train_precision)
        text_file.write('F1 score mean for 10-fold CV: %s \n' %train_f1)
    
    basic_stats(self.tree_clf_rand_ada_new_database, self.X_database_train, self.database)
    basic_stats(self.tree_clf_rand_ada_new_man_add, self.X_man_add_train, self.man_add)
    basic_stats(self.tree_clf_rand_ada_new_transform, self.X_transform_train, self.transform)
    basic_stats(self.tree_clf_rand_ada_new_prot_screen_trans, self.X_prot_screen_trans_train, self.prot_screen_trans)

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

    #maybe I should look at this
    #y_pred = classifier.fit(X_train, y_train).predict(X_test)

    #try out how well the classifier works to predict from the test set
    self.y_pred_class_database = self.tree_clf_rand_ada_new_database.predict(self.X_database_test)
    with open(os.path.join(self.database, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_database_test in y_pred_class_database \n')
    self.y_pred_class_man_add = self.tree_clf_rand_ada_new_man_add.predict(self.X_man_add_test)
    with open(os.path.join(self.man_add, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_man_add_test in y_pred_class_man_add \n')
    self.y_pred_class_transform = self.tree_clf_rand_ada_new_transform.predict(self.X_transform_test)
    with open(os.path.join(self.transform, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_transform_test in y_pred_class_transform \n')
    self.y_pred_class_prot_screen_trans = self.tree_clf_rand_ada_new_prot_screen_trans.predict(self.X_prot_screen_trans_test)
    with open(os.path.join(self.prot_screen_trans, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_prot_screen_trans_test in y_pred_class_prot_screen_trans \n')

    #alternative way to not have to use the test set
    self.y_train_pred_database = cross_val_predict(self.tree_clf_rand_ada_new_database, self.X_database_train, self.y_train, cv=10)
    with open(os.path.join(self.database, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_database_train with 10-fold CV in y_train_pred_database \n')
    self.y_train_pred_man_add = cross_val_predict(self.tree_clf_rand_ada_new_man_add, self.X_man_add_train, self.y_train, cv=10)
    with open(os.path.join(self.man_add, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_man_add_train with 10-fold CV in y_train_pred_man_add \n')
    self.y_train_pred_transform = cross_val_predict(self.tree_clf_rand_ada_new_transform, self.X_transform_train, self.y_train, cv=10)
    with open(os.path.join(self.transform, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_transform_train with 10-fold CV in y_train_pred_transform \n')
    self.y_train_pred_prot_screen_trans = cross_val_predict(self.tree_clf_rand_ada_new_prot_screen_trans, self.X_prot_screen_trans_train, self.y_train, cv=10)
    with open(os.path.join(self.prot_screen_trans, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
      text_file.write('Saving predictions for X_prot_screen_trans_train with 10-fold CV in y_train_pred_prot_screen_trans \n')

    print('*' *80)
    print('*    Calculate prediction stats')
    print('*' *80)

    def prediction_stats(y_test, y_pred_class, directory):
      # calculate accuracy
      y_accuracy = metrics.accuracy_score(self.y_test, y_pred_class)

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

      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Accuracy score or agreement between y_test and y_pred_class: %s \n' %y_accuracy)
        text_file.write('Class distribution for y_test: %s \n' %class_dist)
        text_file.write('Percent 1s in y_test: %s \n' %ones)
        text_file.write('Percent 0s in y_test: %s \n' %zeros)
        text_file.write('Null accuracy in y_test: %s \n' %null_acc)
    
    prediction_stats(self.y_test, self.y_pred_class_database, self.database)
    prediction_stats(self.y_test, self.y_pred_class_man_add, self.man_add)
    prediction_stats(self.y_test, self.y_pred_class_transform, self.transform)
    prediction_stats(self.y_test, self.y_pred_class_prot_screen_trans, self.prot_screen_trans)   

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

    def conf_mat(y_test, y_train, y_pred_class, y_train_pred, directory):
      # IMPORTANT: first argument is true values, second argument is predicted values
      # this produces a 2x2 numpy array (matrix)
      conf_mat_test = metrics.confusion_matrix(y_test, y_pred_class)
      conf_mat_10CV = metrics.confusion_matrix(y_train, y_train_pred)
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
        plt.savefig(os.path.join(directory, 'confusion_matrix_forest_rand_ada_'+name+datestring+'.png'))
        plt.close()

      draw_conf_mat(conf_mat_test, directory, 'test_')
      draw_conf_mat(conf_mat_10CV, directory, 'train_CV_')
      
      TP = conf_mat_test[1, 1]
      TN = conf_mat_test[0, 0]
      FP = conf_mat_test[0, 1]
      FN = conf_mat_test[1, 0]
      
      TP_CV = conf_mat_10CV[1, 1]
      TN_CV = conf_mat_10CV[0, 0]
      FP_CV = conf_mat_10CV[0, 1]
      FN_CV = conf_mat_10CV[1, 0]

      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('confusion matrix using test set: %s \n' %conf_mat_test)
        text_file.write('confusion matrix using 10-fold CV: %s \n' %conf_mat_10CV)
        text_file.write('Slicing confusion matrix for test set into: TP, TN, FP, FN \n')
        text_file.write('Slicing confusion matrix for 10-fold CV into: TP_CV, TN_CV, FP_CV, FN_CV \n')
      
      #calculate accuracy
      acc_score_man_test = (TP + TN) / float(TP + TN + FP + FN)
      acc_score_sklearn_test = metrics.accuracy_score(y_test, y_pred_class)
      acc_score_man_CV = (TP_CV + TN_CV) / float(TP_CV + TN_CV + FP_CV + FN_CV)
      acc_score_sklearn_CV = metrics.accuracy_score(y_train, y_train_pred)  
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Accuracy score: \n')
        text_file.write('accuracy score manual test: %s \n' %acc_score_man_test)
        text_file.write('accuracy score sklearn test: %s \n' %acc_score_sklearn_test)
        text_file.write('accuracy score manual CV: %s \n' %acc_score_man_CV)
        text_file.write('accuracy score sklearn CV: %s \n' %acc_score_sklearn_CV)
        
      #classification error
      class_err_man_test = (FP + FN) / float(TP + TN + FP + FN)
      class_err_sklearn_test = 1 - metrics.accuracy_score(y_test, y_pred_class)
      class_err_man_CV = (FP_CV + FN_CV) / float(TP_CV + TN_CV + FP_CV + FN_CV)
      class_err_sklearn_CV = 1 - metrics.accuracy_score(y_train, y_train_pred)
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Classification error: \n')  
        text_file.write('classification error manual test: %s \n' %class_err_man_test)
        text_file.write('classification error sklearn test: %s \n' %class_err_sklearn_test)
        text_file.write('classification error manual CV: %s \n' %class_err_man_CV)
        text_file.write('classification error sklearn CV: %s \n' %class_err_sklearn_CV)
        
      #sensitivity/recall/true positive rate; correctly placed positive cases  
      sensitivity_man_test = TP / float(FN + TP)
      sensitivity_sklearn_test = metrics.recall_score(y_test, y_pred_class)
      sensitivity_man_CV = TP_CV / float(FN_CV + TP_CV)
      sensitivity_sklearn_CV = metrics.recall_score(y_train, y_train_pred)
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Sensitivity/Recall/True positives: \n')
        text_file.write('sensitivity manual test: %s \n' %sensitivity_man_test)
        text_file.write('sensitivity sklearn test: %s \n' %sensitivity_sklearn_test)
        text_file.write('sensitivity manual CV: %s \n' %sensitivity_man_CV)
        text_file.write('sensitivity sklearn CV: %s \n' %sensitivity_sklearn_CV)
      
      #specificity  
      specificity_man_test = TN / (TN + FP)
      specificity_man_CV = TN_CV / (TN_CV + FP_CV)
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Specificity: \n')
        text_file.write('specificity manual test: %s \n' %specificity_man_test)
        text_file.write('specificity manual CV: %s \n' %specificity_man_CV)
      
      #false positive rate  
      false_positive_rate_man_test = FP / float(TN + FP)
      false_positive_rate_man_CV = FP_CV / float(TN_CV + FP_CV)
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('False positive rate or 1-specificity: \n')
        text_file.write('false positive rate manual test: %s \n' %false_positive_rate_man_test)
        text_file.write('1 - specificity test: %s \n' %(1 - specificity_man_test))
        text_file.write('false positive rate manual CV: %s \n' %false_positive_rate_man_CV)
        text_file.write('1 - specificity CV: %s \n' %(1 - specificity_man_CV))
      
      #precision/confidence of placement  
      precision_man_test = TP / float(TP + FP)
      precision_sklearn_test = metrics.precision_score(y_test, y_pred_class)
      precision_man_CV = TP_CV / float(TP_CV + FP_CV)
      precision_sklearn_CV = metrics.precision_score(y_train, y_train_pred)
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Precision or confidence of classification: \n')
        text_file.write('precision manual: %s \n' %precision_man_test)
        text_file.write('precision sklearn: %s \n' %precision_sklearn_test)
        text_file.write('precision manual CV: %s \n' %precision_man_CV)
        text_file.write('precision sklearn CV: %s \n' %precision_sklearn_CV)
      
      #F1 score; uses precision and recall  
      f1_score_sklearn_test = f1_score(y_test, y_pred_class)
      f1_score_sklearn_CV = f1_score(y_train, y_train_pred)
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('F1 score: \n')
        text_file.write('F1 score sklearn test: %s \n' %f1_score_sklearn_test)
        text_file.write('F1 score sklearn CV: %s \n' %f1_score_sklearn_CV)
        
    conf_mat(self.y_test, self.y_train, self.y_pred_class_database, self.y_train_pred_database, self.database)
    conf_mat(self.y_test, self.y_train, self.y_pred_class_man_add, self.y_train_pred_man_add, self.man_add)
    conf_mat(self.y_test, self.y_train, self.y_pred_class_transform, self.y_train_pred_transform, self.transform)
    conf_mat(self.y_test, self.y_train, self.y_pred_class_prot_screen_trans, self.y_train_pred_prot_screen_trans, self.prot_screen_trans)

    def prediction_probas(forest, X_train, y_train, X_test, y_test, directory, kind): 
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')      
      #probabilities of predicting y_train with X_transform_train using 10-fold CV
      self.y_pred_proba_train_CV = cross_val_predict(forest, X_train, y_train, cv=10, method='predict_proba')
      #probabilities of predicting y_test with X_transform_test
      self.y_pred_proba_test = forest.predict_proba(X_test)
      #self.y_scores=self.forest_clf_rand_new.predict_proba(self.X_transform_train)#train set
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Storing prediction probabilities for X_%s_train and y_train with 10-fold CV in y_pred_proba_train_CV \n' %kind)
        text_file.write('Storing prediction probabilities for X_%s_test and y_test in y_pred_proba_test \n' %kind)
        text_file.write('Plotting histogram for y_pred_proba_train_CV \n')
        text_file.write('Plotting histogram for y_pred_proba_test \n')
      #plot histograms of probabilities  
      def plot_hist_pred_proba(y_pred_proba, name, directory):
        plt.hist(y_pred_proba, bins=8)
        plt.xlim(0,1)
        plt.title('Histogram of predicted probabilities for y_pred_proba_%s to be class 1' %name)
        plt.xlabel('Predicted probability of EP_success')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(directory, 'hist_pred_proba_forest_rand_ada_'+name+datestring+'.png'))
        plt.close()

      plot_hist_pred_proba(self.y_pred_proba_train_CV[:, 1], 'train_CV_', directory)
      plot_hist_pred_proba(self.y_pred_proba_test[:, 1], 'test_', directory)
      
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Getting y_scores for y_pred_proba_train_CV and y_pred_proba_test as y_scores_train_CV and y_scores_test\n')

      #store the predicted probabilities for class 1
      self.y_scores_train_CV = self.y_pred_proba_train_CV[:, 1]
      self.y_scores_test = self.y_pred_proba_test[:, 1]

      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Plotting Precision-Recall for y_test and y_scores_test \n')
        text_file.write('Plotting Precision-Recall for y_train and y_scores_train_CV \n')
      
      #plot precision and recall curve
      def plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest, name, directory):
        plt.plot(thresholds_forest, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds_forest, recalls[:-1], "g--", label="Recall")
        plt.title('Precsion-Recall plot for for EP_success classifier using %s set' %name)
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0,1])
        plt.savefig(os.path.join(directory, 'Precision_Recall_forest_rand_ada_'+name+datestring+'.png'))
        plt.close()

      #plot Precision Recall Threshold curve for test set 
      precisions, recalls, thresholds_forest = precision_recall_curve(self.y_test, self.y_scores_test)
      plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest, 'test_', directory)
      #plot Precision Recall Threshold curve for CV train set 
      precisions, recalls, thresholds_forest = precision_recall_curve(self.y_train, self.y_scores_train_CV)
      plot_precision_recall_vs_threshold(precisions, recalls, thresholds_forest, 'train_CV_', directory)

      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('Plotting ROC curve for y_test and y_scores_test \n')
        text_file.write('Plotting ROC curve for y_train and y_scores_train_CV \n')

      #IMPORTANT: first argument is true values, second argument is predicted probabilities
      #we pass y_test and y_pred_prob
      #we do not use y_pred_class, because it will give incorrect results without generating an error
      #roc_curve returns 3 objects fpr, tpr, thresholds
      #fpr: false positive rate
      #tpr: true positive rate
    
      #plot ROC curves
      def plot_roc_curve(fpr, tpr, name, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.title('ROC curve for EP_success classifier using %s set' %name) 
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        plt.savefig(os.path.join(directory, 'ROC_curve_forest_rand_ada_'+name+datestring+'.png'))
        plt.close()
        
      #ROC curve for test set      
      fpr, tpr, thresholds = roc_curve(self.y_test, self.y_scores_test)#test set
      plot_roc_curve(fpr, tpr, 'test_', directory)
      #ROC curve for 10-fold CV train set
      fpr_CV, tpr_CV, thresholds_CV = roc_curve(self.y_train, self.y_scores_train_CV)#CV train set
      plot_roc_curve(fpr_CV, tpr_CV, 'train_CV_', directory)

      #calculate the area under the curve to get the performance for a classifier
      # IMPORTANT: first argument is true values, second argument is predicted probabilities
      AUC_test = metrics.roc_auc_score(self.y_test, self.y_pred_proba_test[:, 1])
      AUC_train_CV = metrics.roc_auc_score(self.y_train, self.y_pred_proba_train_CV[:, 1])
      with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
        text_file.write('AUC for test set: %s \n' %AUC_test)
        text_file.write('AUC for CV train set: %s \n' %AUC_train_CV)

      # define a function that accepts a threshold and prints sensitivity and specificity
      def evaluate_threshold(tpr, fpr, thresholds, threshold, name, directory):
        sensitivity = tpr[thresholds > threshold][-1]
        specificity = 1 - fpr[thresholds > threshold][-1]
        with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
          text_file.write('Sensitivity for %s at threshold %.2f: %s \n' %(name, threshold, sensitivity))
          text_file.write('Specificity for %s at threshold %.2f: %s \n' %(name, threshold, specificity))

      evaluate_threshold(tpr, fpr, thresholds, 0.6, 'test_', directory)    
      evaluate_threshold(tpr, fpr, thresholds, 0.5, 'test_', directory)
      evaluate_threshold(tpr, fpr, thresholds, 0.4, 'test_', directory)
      evaluate_threshold(tpr, fpr, thresholds, 0.3, 'test_', directory)
      evaluate_threshold(tpr, fpr, thresholds, 0.2, 'test_', directory)
      evaluate_threshold(tpr_CV, fpr_CV, thresholds_CV, 0.6, 'train_CV', directory)
      evaluate_threshold(tpr_CV, fpr_CV, thresholds_CV, 0.5, 'train_CV', directory)
      evaluate_threshold(tpr_CV, fpr_CV, thresholds_CV, 0.4, 'train_CV', directory)
      evaluate_threshold(tpr_CV, fpr_CV, thresholds_CV, 0.3, 'train_CV', directory)
      evaluate_threshold(tpr_CV, fpr_CV, thresholds_CV, 0.2, 'train_CV', directory)

    prediction_probas(self.tree_clf_rand_ada_new_database, self.X_database_train, self.y_train, self.X_database_test, self.y_test, self.database, 'database')
    prediction_probas(self.tree_clf_rand_ada_new_man_add, self.X_man_add_train, self.y_train, self.X_man_add_test, self.y_test, self.man_add, 'man_add')
    prediction_probas(self.tree_clf_rand_ada_new_transform, self.X_transform_train, self.y_train, self.X_transform_test, self.y_test, self.transform, 'transform')
    prediction_probas(self.tree_clf_rand_ada_new_prot_screen_trans, self.X_prot_screen_trans_train, self.y_train, self.X_prot_screen_trans_test, self.y_test, self.prot_screen_trans, 'prot_screen_trans')
    
    def scoring_all(forest, X_train, y_train, X_test, y_test, directory):     
      def scoring(forest, X, y, name, directory, cv):
        # calculate cross_val_scores with different scoring functions for test set
        roc_auc = cross_val_score(forest, X, y, cv=cv, scoring='roc_auc').mean()
        accuracy = cross_val_score(forest, X, y, cv=cv, scoring='accuracy').mean()
        recall = cross_val_score(forest, X, y, cv=cv, scoring='recall').mean()
        precision = cross_val_score(forest, X, y, cv=cv, scoring='precision').mean()
        f1 = cross_val_score(forest, X, y, cv=cv, scoring='f1').mean()
        with open(os.path.join(directory, 'randomforest_ada_randomsearch.txt'), 'a') as text_file:
          text_file.write('ROC_AUC for %s: %s \n' %(name, roc_auc))
          text_file.write('Accuracy for %s: %s \n' %(name, accuracy))
          text_file.write('Recall for %s: %s \n' %(name, recall))
          text_file.write('Precision for %s: %s \n' %(name, precision))
          text_file.write('F1 score for %s: %s \n' %(name, f1))

      scoring(forest, X_test, y_test, 'test', directory, cv=None)
      scoring(forest, X_train, y_train, 'train_CV', directory, cv=10)

    scoring_all(self.tree_clf_rand_ada_new_database, self.X_database_train, self.y_train, self.X_database_test, self.y_test, self.database)
    scoring_all(self.tree_clf_rand_ada_new_man_add, self.X_man_add_train, self.y_train, self.X_man_add_test, self.y_test, self.man_add)
    scoring_all(self.tree_clf_rand_ada_new_transform, self.X_transform_train, self.y_train, self.X_transform_test, self.y_test, self.transform)
    scoring_all(self.tree_clf_rand_ada_new_prot_screen_trans, self.X_prot_screen_trans_train, self.y_train, self.X_prot_screen_trans_test, self.y_test, self.prot_screen_trans)

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)

  database, man_add, transform, prot_screen_trans= make_output_folder(args.outdir)

  ###############################################################################

  random_forest_ada_rand_search = RandomForestAdaRandSearch(metrix, database, man_add, transform, prot_screen_trans)

