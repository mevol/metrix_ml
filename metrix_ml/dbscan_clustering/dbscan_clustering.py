# set up environment
# define command line parameters
# define location of input data
# create output directories
# start the class FeatureCorrelations

import argparse
import os

#import matplotlib
#matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

###############################################################################
#
#  define command line arguments
#
###############################################################################

def parse_command_line():
  '''defining the command line input to make it runable'''
  parser = argparse.ArgumentParser(description='various plots for feature analysis')
  
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
#  load the data from CSV file and creating output directory
#
###############################################################################

def load_data(csv_path):
  '''load the raw data as stored in CSV file'''
  return pd.read_csv(csv_path, na_filter=False, skipinitialspace=True, thousands=',')

def make_output_folder(outdir):
  name = os.path.join(outdir, 'dbscan_clustering')
  os.makedirs(name, exist_ok=True)
  return name

###############################################################################
#
#  class to analyse correlations between features
#
###############################################################################


class DBscan(object):
  '''A class to help analyse the data;
  try to identify linear correlations in the data;
  calculate Pearson Correlation Coefficient with and without
  p-values; create a scatter matrix; inout data must not contain
  any strings or NaN values; also remove any columns with 
  categorical data or transform them first; remove any text except column labels'''

  def __init__(self, data, dbscan_clustering):
    self.data = data
    self.dbscan_clustering = dbscan_clustering
    self.prepare_metrix_data()
    self.clustering()
   
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
    print('*    Preparing input dataframe')
    print('*' *80)

    self.X_data = self.data[["mapCC", "res_frag_ratio"]]

    self.X_data = self.X_data.fillna(0)
    
    self.X_data_scaled = StandardScaler().fit_transform(self.X_data)
    
    with open(os.path.join(self.dbscan_clustering, 'dbscan_clustering.txt'), 'a') as text_file:
      text_file.write('Created the following dataframe: X_data \n')
      #text_file.write(str(self.X_data.columns)+'\n')    

    #print(self.X_data_scaled.columns)

################################################################################
#
#  plotting probability density function for each feature to evaluate continous
#  data and its cumulative distribution function
#
################################################################################

  def clustering(self):
    '''Cluster samples to find classes.'''
    print('*' *80)
    print('*    Clustering samples')
    print('*' *80)

    datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    
    db = DBSCAN(eps=0.4, min_samples=5).fit(self.X_data_scaled)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
#    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#    print("Adjusted Rand Index: %0.3f"
#          % metrics.adjusted_rand_score(labels_true, labels))
#    print("Adjusted Mutual Information: %0.3f"
#          % metrics.adjusted_mutual_info_score(labels_true, labels))
#    print("Silhouette Coefficient: %0.3f"
#          % metrics.silhouette_score(self.X_data_scaled, labels))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = self.X_data_scaled[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = self.X_data_scaled[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.savefig(os.path.join(self.dbscan_clustering, 'alldata_clusters_'+datestring+'.png'))
    plt.show()

    
#        plt.savefig(os.path.join(self.k_means_clustering, 'Silhouette_autosharp_clusters_'+str(n_clusters)+'_'+datestring+'.png'))
#        plt.close()
    #plt.show()
        

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  data = load_data(args.input)

  dbscan_clustering = make_output_folder(args.outdir)

  ###############################################################################

  dbscan_clustering = DBscan(data, dbscan_clustering)

