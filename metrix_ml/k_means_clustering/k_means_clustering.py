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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


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
  name = os.path.join(outdir, 'k_means_clustering')
  os.makedirs(name, exist_ok=True)
  return name

###############################################################################
#
#  class to analyse correlations between features
#
###############################################################################


class Kmeans(object):
  '''A class to help analyse the data;
  try to identify linear correlations in the data;
  calculate Pearson Correlation Coefficient with and without
  p-values; create a scatter matrix; inout data must not contain
  any strings or NaN values; also remove any columns with 
  categorical data or transform them first; remove any text except column labels'''

  def __init__(self, data, k_means_clustering):
    self.data = data
    self.k_means_clustering = k_means_clustering
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
    
    with open(os.path.join(self.k_means_clustering, 'k_means_clustering.txt'), 'a') as text_file:
      text_file.write('Created the following dataframe: X_data \n')
      text_file.write(str(self.X_data.columns)+'\n')    

    print(self.X_data.columns)

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
    
    kmeans = KMeans(n_clusters=5, random_state=42).fit(self.X_data)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    
    range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(self.X_data) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(self.X_data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(self.X_data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        with open(os.path.join(self.k_means_clustering, 'k_means_clustering.txt'), 'a') as text_file:
          text_file.write("For n_clusters ="+str(n_clusters)+'\n')   
          text_file.write("The average silhouette_score is :"+str(silhouette_avg)+'\n')    

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(self.X_data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        #colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters).reshape(-1,4)
        #colors = cm.nipy_spectral((cluster_labels.astype(float) / n_clusters)[0])
        #print(2222, colors)
        ax2.scatter(self.X_data.iloc[:, 0], self.X_data.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        #ax2.plot(self.X_data.iloc[:, 0], marker='*', c=np.arange(len(cluster_labels)))

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        #ax2.plot(centers[:, 0], marker='o')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
#            ax2.plot(c[0], marker='$%d$' % i)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature (mapCC)")
        ax2.set_ylabel("Feature space for the 2nd feature (res_frag_ratio)")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    
        plt.savefig(os.path.join(self.k_means_clustering, 'Silhouette_autosharp_clusters_'+str(n_clusters)+'_'+datestring+'.png'))
        plt.close()
    #plt.show()
        

def run():
  args = parse_command_line()
  
  
  ###############################################################################

  #look at the imported data to get an idea what we are working with
  data = load_data(args.input)

  k_means_clustering = make_output_folder(args.outdir)

  ###############################################################################

  k_means_clustering = Kmeans(data, k_means_clustering)

