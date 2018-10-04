###############################################################################
#
#  imports and set up environment
#
###############################################################################
'''Defining the environment for this class'''
import argparse
import pandas as pd
import pylab as pl
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import seaborn as sns
import scikitplot as skplt
import plotly.plotly as py
import plotly.tools as tls
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from datetime import datetime

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
  names = ['database', 'bbbb']
  result = []
  for name in names:
    name = os.path.join(outdir, 'pca', name)
    os.makedirs(name, exist_ok=True)
    result.append(name)
  return result

###############################################################################
#
#  class for ML using random forest with randomised search
#
###############################################################################

class FeatureDecomposition(object):
  '''This class is the doing the actual work in the following steps:
     * define smaller data frames: database, man_add, transform
     * split the data into training and test set
     * setup and run a grid search for best paramaters to define a random forest
     * create a new random forest with best parameters
     * predict on this new random forest with test data and cross-validated training data
     * analyse the predisctions with graphs and stats
  '''
  def __init__(self, metrix, database, bbbb):
    self.metrix=metrix
    self.database=database
    self.prepare_metrix_data()
    self.split_data()   

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
    Output: smaller dataframes; database
    '''
    print('*' *80)
    print('*    Preparing input dataframe metrix_database')
    print('*' *80)

    #look at the data that is coming from processing
    attr_database = ['IoverSigma', 'anomalousslope', 'anomalousCC', 'anomalousmulti', 'multiplicity',
                      'diffI', 'cchalf', 'totalobservations', 'wilsonbfactor', 'lowreslimit',
                      'anomalouscompl', 'highreslimit', 'completeness', 'totalunique', 'RmergediffI',
                      'RmergeI', 'RmeasI', 'RmeasdiffI', 'RpimdiffI', 'RpimI', 'diffF']
    metrix_database = self.metrix[attr_database]

    self.X_database = metrix_database
    
    with open(os.path.join(self.database, 'pca_database.txt'), 'a') as text_file:
      text_file.write('Preparing input data as metrix_database with following attributes %s \n' %(attr_database))
      
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

#normal split of samples    
#    X_database_train, X_database_test, y_train, y_test = train_test_split(self.X_database, y, test_size=0.2, random_state=42)

#stratified split of samples
    X_database_train, X_database_test, y_train, y_test = train_test_split(self.X_database, y, test_size=0.2, random_state=42, stratify=y)

    assert self.X_database.columns.all() == X_database_train.columns.all()
    
    self.X_database_train = X_database_train
    self.X_database_test = X_database_test
    self.y_train = y_train
    self.y_test = y_test
    
    #standardise data
    X_database_train_std = StandardScaler().fit_transform(self.X_database_train)
    self.X_database_train_std = X_database_train_std
    
    def pca_manual(X_train_std):
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')          
      mean_vec = np.mean(X_database_train_std, axis=0)
      cov_mat = (X_database_train_std - mean_vec).T.dot((X_database_train_std - mean_vec)) / (X_database_train_std.shape[0]-1)
      #print('Covariance matrix \n%s' %cov_mat)
      cov_mat = np.cov(X_database_train_std.T)

      eig_vals, eig_vecs = np.linalg.eig(cov_mat)

      with open(os.path.join(self.database, 'pca_database.txt'), 'a') as text_file:
        text_file.write('Eigenvectors \n%s' %eig_vecs)
        text_file.write('\nEigenvalues \n%s' %eig_vals)


#      print('Eigenvectors \n%s' %eig_vecs)
#      print('\nEigenvalues \n%s' %eig_vals)

      for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
      print('Everything ok!')
      
      eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

      # Sort the (eigenvalue, eigenvector) tuples from high to low
      eig_pairs.sort()
      eig_pairs.reverse()

      # Visually confirm that the list is correctly sorted by decreasing eigenvalues
      with open(os.path.join(self.database, 'pca_database.txt'), 'a') as text_file:
        text_file.write('Eigenvalues in descending order:')      
      
#      print('Eigenvalues in descending order:')
      for i in eig_pairs:
        with open(os.path.join(self.database, 'pca_database.txt'), 'a') as text_file:
          text_file.write(str(i[0])+'\n')
        text_file.close() 
#        print(i[0])

      tot = sum(eig_vals)
      var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
      cum_var_exp = np.cumsum(var_exp)

      with open(os.path.join(self.database, 'pca_database.txt'), 'a') as text_file:
        text_file.write('Plotting contributions of each PC in bar plot')
      
      PCs = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8',
             'PC-9', 'PC-10', 'PC-11', 'PC-12', 'PC-13', 'PC-14', 'PC-15',
             'PC-16', 'PC-17', 'PC-18', 'PC-19', 'PC-20', 'PC-21']   
      
      plt.rcdefaults()
      fig, ax = plt.subplots(dpi=244)
      height = var_exp
      y_pos = ['PC %s' %i for i in PCs]
      height2 = cum_var_exp
      ax.bar(y_pos, height, align='center', color='blue')
      ax.plot(y_pos, height2, marker='o', markersize=2,color='orange', label='cumulative explained variance')
      plt.xticks(y_pos, PCs, rotation=90)
      plt.title('Explained variance by different principal components')
      plt.ylabel('Explained variance in percent')
      plt.xlabel('Principal components')
      plt.legend(loc="upper right")
      ax.set_ylim((0, 100))
      plt.grid(True, axis='y', which='major')
      plt.savefig(os.path.join(self.database, 'PCA_results_database_'+datestring+'.png'))     
      plt.close()
#      plt.show()

    pca_manual(self.X_database_train_std)

    def run_pca_pair_one(X, columns):
      '''projecting the data down into 2 dimensions'''
      #fit() just fits the data
      #fit_transform() fits the data and reduces the dimensions
      #pca = PCA(n_components=2)#getting the first 2 PCs
      pca = PCA(n_components=21)#get all 21 PCs
      pca.fit_transform(X)     
      print(pca.components_)
      PCs = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8',
             'PC-9', 'PC-10', 'PC-11', 'PC-12', 'PC-13', 'PC-14', 'PC-15',
             'PC-16', 'PC-17', 'PC-18', 'PC-19', 'PC-20', 'PC-21']
      df = pd.DataFrame(pca.components_, columns=columns, index = PCs)
      
      N=21
      ind = np.arange(N)
      width = 0.35
      p1 = plt.bar(ind, df.iloc[0], width)
      p2 = plt.bar(ind, df.iloc[1], width)
      p3 = plt.bar(ind, df.iloc[2], width)
      p4 = plt.bar(ind, df.iloc[3], width)
      p5 = plt.bar(ind, df.iloc[4], width)
      p6 = plt.bar(ind, df.iloc[5], width)
      p7 = plt.bar(ind, df.iloc[6], width)
      p8 = plt.bar(ind, df.iloc[7], width)
      p9 = plt.bar(ind, df.iloc[8], width)
      p10 = plt.bar(ind, df.iloc[9], width)
      p11 = plt.bar(ind, df.iloc[10], width)
      p12 = plt.bar(ind, df.iloc[11], width)
      p13 = plt.bar(ind, df.iloc[12], width)
      p14 = plt.bar(ind, df.iloc[13], width)
      p15 = plt.bar(ind, df.iloc[14], width)
      p16 = plt.bar(ind, df.iloc[15], width)
      p17 = plt.bar(ind, df.iloc[16], width)
      p18 = plt.bar(ind, df.iloc[17], width)
      p19 = plt.bar(ind, df.iloc[18], width)
      p20 = plt.bar(ind, df.iloc[19], width)
      p21 = plt.bar(ind, df.iloc[20], width)
      
      plt.ylabel('PC contribution')
      plt.title('Feature dominance in each PC')
      plt.xticks(ind, df.columns, rotation=90)
      plt.tight_layout()
      plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0], p13[0], p14[0], p15[0], p16[0], p17[0], p18[0], p19[0], p20[0], p21[0]), ('PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8',
             'PC-9', 'PC-10', 'PC-11', 'PC-12', 'PC-13', 'PC-14', 'PC-15',
             'PC-16', 'PC-17', 'PC-18', 'PC-19', 'PC-20', 'PC-21'), loc="upper right")
      plt.show()

      
      print(df.info())
      print(df.describe())
      print(df.head())
#      PCA(copy=True, iterated_power='auto', random_state=42, svd_solver='auto')
      print(pca.explained_variance_ratio_)
      print(pca.singular_values_)
      
#    grid_search_man_add = grid_search.fit(self.X_man_add_train, self.y_train)
#    with open(os.path.join(self.man_add_features, 'randomforest_gridsearch.txt'), 'a') as text_file:
#      text_file.write('Best parameters: ' +str(grid_search_man_add.best_params_)+'\n')
#      text_file.write('Best score: ' +str(grid_search_man_add.best_score_)+'\n')
      
      
      
      with open(os.path.join(self.database, 'pca_database.txt'), 'a') as text_file:
        text_file.write('Feature contribution for each PC \n')
        text_file.write(str(pca.components_)+'\n')
      #df.to_csv((os.path.join(self.database, 'pca_database.txt'), sep=' ', mode='a')  
        #text_file.write(df +'\n')
      with open(os.path.join(self.database, 'pca_database.txt'), 'a') as text_file:
        text_file.write('explained variance ratio: '+str(pca.explained_variance_ratio_)+'\n')
        text_file.write('Singular values: '+str(pca.singular_values_)+'\n')
      
      
    run_pca_pair_one(self.X_database_train, self.X_database_train.columns)     







##########################################################################
    
    #metrix_database = StandardScaler().fit_transform(metrix_database)
    
    #metrix_database = pd.DataFrame(preprocessing.scale(metrix_database),columns = self.metrix[attr_database]) 

    
    #print(metrix_database.head())
      
    self.X_database = metrix_database
    
    #print(self.X_database.head())

    with open(os.path.join(self.database, 'pca_database.txt'), 'a') as text_file:
      text_file.write('Created the following dataframes: metrix_database \n')
      
    def run_pca_pair_one(X, columns):
      '''projecting the data down into 2 dimensions'''
      #fit() just fits the data
      #fit_transform() fits the data and reduces the dimensions
      #pca = PCA(n_components=2)#getting the first 2 PCs
      pca = PCA(n_components=21)#get all 21 PCs
      pca.fit_transform(X)
      print(pca.components_)
      PCs = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8',
             'PC-9', 'PC-10', 'PC-11', 'PC-12', 'PC-13', 'PC-14', 'PC-15',
             'PC-16', 'PC-17', 'PC-18', 'PC-19', 'PC-20', 'PC-21']
      df = pd.DataFrame(pca.components_, columns=columns, index = PCs)
      print(df.head())
#      PCA(copy=True, iterated_power='auto', random_state=42, svd_solver='auto')
      print(pca.explained_variance_ratio_)
      print(pca.singular_values_)      
      
    run_pca_pair_one(self.X_database_train, self.X_database_train.columns)     
    
    def run_pca_reduced(X):
      pca = PCA(n_components=0.95)#calculate the PCs to explain 95% of variance
      X_reduced = pca.fit_transform(X)
      print(pca.explained_variance_ratio_)
      print(pca.singular_values_)      
      
    run_pca_reduced(self.X_database)

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)
  
  database, bbbb= make_output_folder(args.outdir)

  ###############################################################################

  feature_decomposition = FeatureDecomposition(metrix, database, bbbb)

