'''
Created on 24 Jun 2020

@author: aliv
'''

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

from .correction.pbm_clustering_correction import get_position_separated_ctr_lists
from .click_simulation.base import load_clicks_pickle

def prepare_for_plot(data, bw = None):
  kde = stats.gaussian_kde(data, bw_method=bw)
  if bw == None:
    bw = getattr(kde, "scotts_factor")() * np.std(data)
  grid = sns.utils._kde_support(data, bw, 300, 3, [-np.inf,np.inf])
#   plt.plot(grid, np.minimum(15*np.ones_like(grid), kde(grid)),label='combined')
  return grid, kde(grid), bw

def plot_clicks(clicks_pickle_path, topk, positions):
  clicks, _, doclist_ranges = load_clicks_pickle(clicks_pickle_path)
  
  ctr_lists = get_position_separated_ctr_lists(clicks, doclist_ranges, topk, False)
  clip_y = 15
  kde = stats.gaussian_kde(ctr_lists[0][:,0]/ctr_lists[0][:,1])
  bw = getattr(kde, "scotts_factor")()/2. #* np.std(ctr_lists[0][:,0]/ctr_lists[0][:,1])

  for i, pos in enumerate(positions):
    x, y, bw = prepare_for_plot(ctr_lists[pos][:,0]/ctr_lists[pos][:,1], bw)
    axs = plt.subplot(len(positions),1,i+1)
    plt.xlim(0.,1.)
    plt.ylim(-0.01,12)
    plt.plot(x, np.minimum(clip_y*np.ones_like(x),y))
    
  plt.show()
  
  
def plot_two_clicks(clicks_pickle_path, clicks_pickle_path2, topk, max_position_plot):
  clicks, _, doclist_ranges = load_clicks_pickle(clicks_pickle_path)
  clicks2, _, doclist_ranges2 = load_clicks_pickle(clicks_pickle_path2)
  
  ctr_lists = get_position_separated_ctr_lists(clicks, doclist_ranges, topk, False)
  ctr_lists2 = get_position_separated_ctr_lists(clicks2, doclist_ranges2, topk, False)
  clip_y = 10
  kde = stats.gaussian_kde(ctr_lists[0][:,0]/ctr_lists[0][:,1])
  bw = getattr(kde, "scotts_factor")()/2. #* np.std(ctr_lists[0][:,0]/ctr_lists[0][:,1])
  kde2 = stats.gaussian_kde(ctr_lists2[0][:,0]/ctr_lists2[0][:,1])
  bw2 = getattr(kde2, "scotts_factor")()/2. #* np.std(ctr_lists[0][:,0]/ctr_lists[0][:,1])

  for i in range(max_position_plot):
    x, y, bw = prepare_for_plot(ctr_lists[i][:,0]/ctr_lists[i][:,1], bw)
    x2, y2, bw2 = prepare_for_plot(ctr_lists2[i][:,0]/ctr_lists2[i][:,1], bw2)
    axs = plt.subplot(max_position_plot,1,i+1)
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,clip_y + 0.5)
    plt.plot(x, np.minimum(clip_y*np.ones_like(x),y), label='graded relevance')
    plt.plot(x2, np.minimum(clip_y*np.ones_like(x2),y2), label='binary relevance')
    
  plt.legend()
  plt.show()
  
  
def plot_multiple_separate_clicks(clicks_pickle_paths, topk, position_plots):
  plt.subplots(len(clicks_pickle_paths), len(position_plots),figsize=(16,9))
  for click_path_ind, clicks_pickle_path in enumerate(clicks_pickle_paths):
    clicks, _, doclist_ranges = load_clicks_pickle(clicks_pickle_path)
    
    sess_cnt = 0
    for i in range(len(clicks)):
      sess_cnt += len(clicks[i])
    sess_cnt /= len(clicks)
    ctr_lists = get_position_separated_ctr_lists(clicks, doclist_ranges, topk, False)
    clip_y = 7
    kde = stats.gaussian_kde(ctr_lists[0][:,0]/ctr_lists[0][:,1])
    bw = getattr(kde, "scotts_factor")()/1.2 #* np.std(ctr_lists[0][:,0]/ctr_lists[0][:,1])
  
    for i, pos in enumerate(position_plots):
      x, y, bw = prepare_for_plot(ctr_lists[pos][:,0]/ctr_lists[pos][:,1], bw)
      axs = plt.subplot(len(clicks_pickle_paths), len(position_plots),click_path_ind * len(position_plots) + i+1)
      if i == 0:
        plt.ylabel('{} sessions'.format(int(sess_cnt)), fontsize=16)
      else:
        plt.gca().axes.yaxis.set_ticklabels([])
      if click_path_ind == 0:
        plt.title('position {}'.format(pos+1), fontsize=16)
      if click_path_ind < len(clicks_pickle_paths) - 1:
        plt.gca().axes.xaxis.set_ticklabels([])
      plt.subplots_adjust(wspace=0.05, hspace=0.05)
      plt.xlim(-0.1,0.2)
      plt.ylim(-0.1,clip_y + 0.5)
      plt.plot(x, np.minimum(clip_y*np.ones_like(x),y))
    
#   plt.show()
#   plt.savefig('/Users/aliv/Dropbox/MyPapers/slides-2020-clustering-cltr-soos/figure/dist.pdf',bbox_inches='tight', pad_inches=0)
  