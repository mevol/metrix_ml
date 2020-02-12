###############################################################################
#
#  imports and set up environment
#
###############################################################################
'''Defining the environment for this class'''
import argparse
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import pylab as pl
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import seaborn as sns
import scikitplot as skplt
#import plotly.plotly as py
#import plotly.tools as tls
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
  parser = argparse.ArgumentParser(description='Principal COmponent Analysis')
  
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
  output_dir = os.path.join(outdir, 'pca')
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

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
  def __init__(self, metrix, output_dir):
    self.metrix = metrix
    self.output_dir = output_dir
    self.prepare_metrix_data()
    self.split_data()
    self.run_pca()

###############################################################################
#
#  creating smaller data frame
#
###############################################################################

  def prepare_metrix_data(self):
    '''Function to create smaller dataframe with columns of interest.
    ******
    Input: large data frame
    Output: smaller dataframe
    '''
    print('*' *80)
    print('*    Preparing input dataframe X_metrix')
    print('*' *80)

    #database plus manually added data
    self.X_metrix = self.metrix[['IoverSigma', 'completeness', 'RmergeI',
                    'lowreslimit', 'RpimI', 'multiplicity', 'RmeasdiffI',
                    'wilsonbfactor', 'RmeasI', 'highreslimit', 'RpimdiffI', 
                    'RmergediffI', 'totalobservations', 'cchalf', 'totalunique',
                    'mr_reso', 'eLLG', 'tncs', 'seq_ident', 'model_res',
                    'No_atom_chain', 'MW_chain', 'No_res_chain', 'No_res_asu',
                    'likely_sg_no', 'xia2_cell_volume', 'Vs', 'Vm',
                    'No_mol_asu', 'MW_asu', 'No_atom_asu']]

    self.X_metrix = self.X_metrix.fillna(0)

    with open(os.path.join(self.output_dir,
              'pca.txt'), 'a') as text_file:
      text_file.write('Created dataframe X_metrix \n')
      text_file.write('with columns: \n')
      text_file.write(str(self.X_metrix.columns)+ '\n')
      
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
    print('*' *80)
    print('*    Splitting data into test and training set with test=20%')
    print('*' *80)

    y = self.metrix['MR_success']

#stratified split of samples
    X_metrix_train, X_metrix_test, y_train, y_test = train_test_split(self.X_metrix, y, test_size=0.2, random_state=42, stratify=y)
    
    assert self.X_metrix.columns.all() == X_metrix_train.columns.all()

    self.X_metrix_train = X_metrix_train
    self.X_metrix_test = X_metrix_test
    self.y_train = y_train
    self.y_test = y_test

    with open(os.path.join(self.output_dir,
              'pca.txt'), 'a') as text_file:
      text_file.write('Spliting into training and test set 80-20 \n')
      text_file.write('X_metrix: X_metrix_train, X_metrix_test \n')
      text_file.write('y(MR_success): y_train, y_test \n')

###############################################################################
    
    #standardise data
    X_metrix_train_std = StandardScaler().fit_transform(self.X_metrix_train)
    self.X_metrix_train_std = X_metrix_train_std

###############################################################################
    
  def run_pca(self):  
    def pca_manual(X_train_std):
      print('*' *80)
      print('*    Running manual PCA')
      print('*' *80)

      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')          
      mean_vec = np.mean(X_train_std, axis=0)
      cov_mat = (X_train_std - mean_vec).T.dot((X_train_std - mean_vec)) / (X_train_std.shape[0]-1)
      #print('Covariance matrix \n%s' %cov_mat)
      cov_mat = np.cov(X_train_std.T)

      eig_vals, eig_vecs = np.linalg.eig(cov_mat)

      with open(os.path.join(self.output_dir, 'pca.txt'), 'a') as text_file:
        text_file.write('Eigenvectors \n%s' %eig_vecs)
        text_file.write('\nEigenvalues \n%s' %eig_vals)

      for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev), decimal=2)
      
      eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

      # Sort the (eigenvalue, eigenvector) tuples from high to low
      eig_pairs.sort()
      eig_pairs.reverse()

      # Visually confirm that the list is correctly sorted by decreasing eigenvalues
      with open(os.path.join(self.output_dir, 'pca.txt'), 'a') as text_file:
        text_file.write('Eigenvalues in descending order:')      
      
      #print('Eigenvalues in descending order:')
      for i in eig_pairs:
        with open(os.path.join(self.output_dir, 'pca.txt'), 'a') as text_file:
          text_file.write(str(i[0])+'\n')
        text_file.close() 
#        print(i[0])

      tot = sum(eig_vals)
      var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
      cum_var_exp = np.cumsum(var_exp)

      with open(os.path.join(self.output_dir, 'pca.txt'), 'a') as text_file:
        text_file.write('Plotting cumulative explained variance in bar plot')

      print('*' *80)
      print('*    Plotting cumulative explained variance')
      print('*' *80)
    
      PCs = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8',
             'PC-9', 'PC-10', 'PC-11', 'PC-12', 'PC-13', 'PC-14', 'PC-15',
             'PC-16', 'PC-17', 'PC-18', 'PC-19', 'PC-20', 'PC-21', 'PC-22',
             'PC-23', 'PC-24', 'PC-25', 'PC-26', 'PC-27', 'PC-28', 'PC-29',
             'PC-30', 'PC-31']   
      
      plt.rcdefaults()
      fig, ax = plt.subplots(dpi=600)
      height = var_exp
      y_pos = ['PC %s' %i for i in PCs]
      height2 = cum_var_exp
      ax.bar(y_pos, height, align='center', color='blue')
      ax.plot(y_pos, height2, marker='o', markersize=2,color='orange',
              label='cumulative explained variance')
      plt.xticks(y_pos, PCs, rotation=90)
      plt.yticks(np.arange(0, 100, step=10))
      plt.title('Explained variance by different principal components')
      plt.ylabel('Explained variance in percent')
      plt.xlabel('Principal components')
      plt.legend(loc="upper right")
      ax.set_ylim((0, 100))
      plt.grid(True, axis='y', which='both')
      plt.savefig(os.path.join(self.output_dir,
              'PCA_cumulative_explained_variance_transform_'+datestring+'.png'))     
      plt.close()

    pca_manual(self.X_metrix_train_std)
    
###############################################################################
    
    def run_pca_reduced(X_train_std):
      print('*' *80)
      print('*    Running PCA to get 95% coverage')
      print('*' *80)

      #fit() just fits the data
      #fit_transform() fits the data and reduces the dimensions

      pca = PCA(n_components=0.95)#calculate the PCs to explain 95% of variance
      X_train_std_reduced = pca.fit_transform(X_train_std)
      coverage = pca.explained_variance_ratio_
      with open(os.path.join(self.output_dir, 'pca.txt'), 'a') as text_file:
        text_file.write('PCs necessary to get 95% coverage: ' +str(coverage)+'\n')
      max_num_pc = len(pca.explained_variance_ratio_)
      #print(max_num_pc)
      #print(pca.singular_values_)   
      return max_num_pc
      
    max_num_pc = run_pca_reduced(self.X_metrix_train_std)

###############################################################################

    def pca_results(X_train_std, columns, num_pc):
      print('*' *80)
      print('*    Analysis for number of PCs necessary to get 95% coverage')
      print('*' *80)

      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      pca = PCA(n_components=num_pc)#for all PCs needed to get 95%
      pca.fit_transform(X_train_std)

      components = pca.components_
      with open(os.path.join(self.output_dir, 'pca.txt'), 'a') as text_file:
        text_file.write('PCA components: ' +str(components)+'\n')
#      print(pca.components_)

      exp_var = pca.explained_variance_
      with open(os.path.join(self.output_dir, 'pca.txt'), 'a') as text_file:
        text_file.write('Explained variance: ' +str(exp_var)+'\n')     
#      print(pca.explained_variance_)
      
      exp_var_ratio = pca.explained_variance_ratio_
      with open(os.path.join(self.output_dir, 'pca.txt'), 'a') as text_file:
        text_file.write('Explained variance ratio: ' +str(exp_var_ratio)+'\n')          
#      print(pca.explained_variance_ratio_)
      
      sing_val = pca.singular_values_
      with open(os.path.join(self.output_dir, 'pca.txt'), 'a') as text_file:
        text_file.write('Singular values: ' +str(sing_val)+'\n')           
#      print(pca.singular_values_)
    
     # Dimension indexing
      dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
      # PCA components
      components = pd.DataFrame(np.round(pca.components_, num_pc), columns = columns) 
      components.index = dimensions

      # PCA explained variance
      ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1) 
      variance_ratios = pd.DataFrame(np.round(ratios, num_pc), columns = ['Explained Variance']) 
      variance_ratios.index = dimensions

      # Create a bar plot visualization
      fig, ax = plt.subplots(figsize = (14,8))

      # Plot the feature weights as a function of the components
      print('*' *80)
      print('*    Plotting feature weights for PCs necessary to get 95% coverage')
      print('*' *80)

      components.plot(ax = ax, kind = 'bar')
      ax.set_ylabel("Feature Weights") 
      ax.set_xticklabels(dimensions, rotation=90)
      plt.legend(loc='best')
      plt.tight_layout()
      plt.savefig(os.path.join(self.output_dir,
                  'Feature_weights_per_PC_transform_'+datestring+'.png'))
      plt.close()
        
      # Display the explained variance ratios# 
      for i, ev in enumerate(pca.explained_variance_ratio_): 
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f"%(ev))

      # Return a concatenated DataFrame
      return pd.concat([variance_ratios, components], axis = 1)

    pca_results = pca_results(self.X_metrix_train_std,
                              self.X_metrix_train.columns,
                              max_num_pc)
    #print(pca_results)

###############################################################################

    def run_pca_pair_one(X_train_std, columns):
      print('*' *80)
      print('*    Plotting biplot for first two PCs')
      print('*' *80)

      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      pca = PCA(n_components=2).fit(X_train_std)
      reduced_X_train_std = pca.transform(X_train_std)
      pca_samples = pca.transform(X_train_std)
      reduced_X_train_std = pd.DataFrame(reduced_X_train_std,
                                       columns = ['Dimension 1', 'Dimension 2'])

      def biplot(X_train_std, reduced_X_train_std, pca, columns):
    
        fig, ax = plt.subplots(figsize = (14,8), dpi=600)
    
        # scatterplot of the reduced data 
        ax.scatter(x=reduced_X_train_std.loc[:, 'Dimension 1'],
                   y=reduced_X_train_std.loc[:, 'Dimension 2'],
                   facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
        feature_vectors = pca.components_.T

        # using scaling factors to make the arrows
        arrow_size, text_pos = 7.0, 8.0,

        # projections of the original features
        for i, v in enumerate(feature_vectors):
          ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], head_width=0.2,
                   head_length=0.2, linewidth=2, color='red')
          ax.text(v[0]*text_pos, v[1]*text_pos, columns[i], color='black',
                  ha='center', va='center', fontsize=18)

        ax.set_xlabel("Dimension 1", fontsize=14)
        ax.set_ylabel("Dimension 2", fontsize=14)
        ax.set_title("PC plane with original feature projections.", fontsize=16)
        plt.savefig(os.path.join(self.output_dir,
                    'PC_biplot_PC1_PC2_transform_'+datestring+'.png'))
        plt.close()
        return ax

      biplot(X_train_std, reduced_X_train_std, pca, columns)

    run_pca_pair_one(self.X_metrix_train_std,
                     self.X_metrix_train.columns)
    
###############################################################################    

    def feature_weights(X_train_std, columns, num_pc):
      '''projecting the data down into 2 dimensions'''
      datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
      pca = PCA(n_components=num_pc)
      pca.fit_transform(X_train_std)
      dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
             
      df = pd.DataFrame(pca.components_, columns=columns, index = dimensions)     
      df_i = df.T
      lab_i = df_i.columns
      N_i = len(lab_i)
      ind_i = np.arange(N_i)
      width = 0.5

      plt.figure(figsize=(45,30))

      p1_i = plt.bar(ind_i, df_i.iloc[0], width, color='red')
      p2_i = plt.bar(ind_i, df_i.iloc[1], width, color='green')
      p3_i = plt.bar(ind_i, df_i.iloc[2], width, color='blue')
      p4_i = plt.bar(ind_i, df_i.iloc[3], width, color='yellow')
      p5_i = plt.bar(ind_i, df_i.iloc[4], width, color='black')
      p6_i = plt.bar(ind_i, df_i.iloc[5], width, color='cyan')
      p7_i = plt.bar(ind_i, df_i.iloc[6], width, color='darkorange')
      p8_i = plt.bar(ind_i, df_i.iloc[7], width, color='lightcoral')
      p9_i = plt.bar(ind_i, df_i.iloc[8], width, color='gray')
      p10_i = plt.bar(ind_i, df_i.iloc[9], width, color='powderblue')
      p11_i = plt.bar(ind_i, df_i.iloc[10], width, color='darkmagenta')
      p12_i = plt.bar(ind_i, df_i.iloc[11], width, color='lavender')
      p13_i = plt.bar(ind_i, df_i.iloc[12], width, color='wheat')
      p14_i = plt.bar(ind_i, df_i.iloc[13], width, color='mediumpurple')
      p15_i = plt.bar(ind_i, df_i.iloc[14], width, color='sandybrown')
      p16_i = plt.bar(ind_i, df_i.iloc[15], width, color='lawngreen')
      p17_i = plt.bar(ind_i, df_i.iloc[16], width, color='plum')
      p18_i = plt.bar(ind_i, df_i.iloc[17], width, color='red', hatch="/")
      p19_i = plt.bar(ind_i, df_i.iloc[18], width, color='green', hatch="/")
      p20_i = plt.bar(ind_i, df_i.iloc[19], width, color='blue', hatch="/")
      p21_i = plt.bar(ind_i, df_i.iloc[20], width, color='yellow', hatch="/")
      p22_i = plt.bar(ind_i, df_i.iloc[21], width, color='cyan', hatch="/")
      p23_i = plt.bar(ind_i, df_i.iloc[22], width, color='darkorange', hatch="/")
      p24_i = plt.bar(ind_i, df_i.iloc[23], width, color='lightcoral', hatch="/")
      p25_i = plt.bar(ind_i, df_i.iloc[24], width, color='gray', hatch="/")
      p26_i = plt.bar(ind_i, df_i.iloc[25], width, color='darkmagenta', hatch="/")
      p27_i = plt.bar(ind_i, df_i.iloc[26], width, color='lavender', hatch="/")
      p28_i = plt.bar(ind_i, df_i.iloc[27], width, color='wheat', hatch="/")
      p29_i = plt.bar(ind_i, df_i.iloc[28], width, color='mediumpurple', hatch="/")
      p30_i = plt.bar(ind_i, df_i.iloc[29], width, color='sandybrown', hatch="/")
      p31_i = plt.bar(ind_i, df_i.iloc[30], width, color='lawngreen', hatch="/")
#      p32_i = plt.bar(ind_i, df_i.iloc[31], width, color='plum', hatch="/")
#      p33_i = plt.bar(ind_i, df_i.iloc[32], width, color='red', hatch="*")
#      p34_i = plt.bar(ind_i, df_i.iloc[33], width, color='green', hatch="*")
#      p35_i = plt.bar(ind_i, df_i.iloc[34], width, color='blue', hatch="*")
#      p36_i = plt.bar(ind_i, df_i.iloc[35], width, color='yellow', hatch="*")
#      p37_i = plt.bar(ind_i, df_i.iloc[36], width, color='cyan', hatch="*")
#      p38_i = plt.bar(ind_i, df_i.iloc[37], width, color='darkorange', hatch="*")
#      p39_i = plt.bar(ind_i, df_i.iloc[38], width, color='lightcoral', hatch="*")
#      p40_i = plt.bar(ind_i, df_i.iloc[39], width, color='gray', hatch="*")
#      p41_i = plt.bar(ind_i, df_i.iloc[40], width, color='darkmagenta', hatch="*")
#      p42_i = plt.bar(ind_i, df_i.iloc[41], width, color='lavender', hatch="*")
           
      plt.title('Feature dominance in each PC', fontsize=20)
      plt.xlabel('Number of PCs', fontsize=20)
      plt.ylabel('Contribution of individual features in PC', fontsize=20)
      plt.xticks(ind_i, lab_i, rotation=90, fontsize=20)
      plt.legend(labels=df_i.index, loc='best', fontsize=20)
      plt.tight_layout()
      plt.savefig(os.path.join(self.output_dir,
                  'PC_feature_contribution_transform_'+datestring+'.png'))     
      plt.close()
      
    feature_weights(self.X_metrix_train_std,
                    self.X_metrix_train.columns,
                    max_num_pc)     

def run():
  args = parse_command_line()
  
  
###############################################################################

  #look at the imported data to get an idea what we are working with
  metrix = load_metrix_data(args.input)
  
  output_dir = make_output_folder(args.outdir)

###############################################################################

  feature_decomposition = FeatureDecomposition(metrix, output_dir)

