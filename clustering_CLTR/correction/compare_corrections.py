'''
Created on 16 Oct 2020

@author: aliv
'''
import numpy as np
from .pbm_affine_correction import pad_and_reshape
from .pbm_clustering_correction import get_position_separated_ctr_lists

import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl

import pickle

def compare_with_gold(data, corrections, position, num_queries):
  correcteds = {}
  topk = 50
  for correction in corrections:
    corrections[correction].correct()
    correcteds[correction] = pad_and_reshape(corrections[correction].doclist_ranges['train'], topk, corrections[correction].corrected_clicks['train'])
  rel = pad_and_reshape(corrections[correction].doclist_ranges['train'], topk, data.train.label_vector[corrections[correction].argsorted['train']])
  errors = np.zeros([50,10])
  for pos in range(topk):
    x = np.zeros_like(rel[:,pos])
    x[rel[:,pos] > 2] = 1.
    errors[pos,0] = np.sum(x)
    for i, cor in enumerate(correcteds):
      y = correcteds[cor][:,pos]
      by = np.zeros_like(y)
      by[y > np.max(y)/2] = 1
      err = x-by
      errors[pos,3*i+1:3*i+4] = [len(err[err==1]), len(err[err==-1]), np.sum(np.abs(err))]
      print('position {} of {} :rel error = {}, nonrel error = {}, total = {}'.format(pos, cor, len(err[err==1]), len(err[err==-1]), np.sum(np.abs(err))))
      
  print(errors)
  for pos in range(position):
    print('position: {}'.format(pos))
    print('rel',end='\t')
    for correction in corrections:
      print(correction, end='\t')
    print()
    for q in range(num_queries):
      cor_str = ''
      for corrected in correcteds:
        cor_str += '{}\t'.format(correcteds[corrected][q, pos])
      print('{}\t{}'.format(rel[q, pos], cor_str))
      
      
def get_unique_xy(x,y):
  unique_x, unique_index = np.unique(x,return_index=True)
  return unique_x, y[unique_index]

def plot_corrected(data, corrections, pos_list, prefix):
  correcteds = {}

  _G_COLORS_0 = ['#f4a582', '#b8e186', '#92c5de', '#b2abd2']
  _G_COLORS = ['#ca0020', '#4dac26', '#0571b0', '#5e3c99']
  topk = 20
  for corrections_pointer, correction in enumerate(corrections):
    corrections[correction].correct()
    correcteds[correction] = pad_and_reshape(corrections[correction].doclist_ranges['train'], topk, corrections[correction].corrected_clicks['train'])
  rel_g = pad_and_reshape(corrections[correction].doclist_ranges['train'], topk, data.train.label_vector[corrections[correction].argsorted['train']])
  rel_b = np.zeros_like(rel_g)
  rel_b[rel_g > 2] = 1
  legend_info = {'pbm_affine_1':{'label':'AC (Relevant)', 'color':_G_COLORS[1]}, 'pbm_soft_clustering_1':{'label':'MBC (Relevant)', 'color':_G_COLORS[0]},
                 'pbm_affine_0':{'label':'AC (Non-relevant)', 'color':_G_COLORS_0[1]}, 'pbm_soft_clustering_0':{'label':'MBC (Non-relevant)', 'color':_G_COLORS_0[0]},
                 'pbm_soft_clustering_b_1':{'label':'MBC Binomial (Relevant)', 'color':_G_COLORS[2]}, 'pbm_soft_clustering_g_1':{'label':'MBC Gaussian (Relevant)', 'color':_G_COLORS[0]},
                 'pbm_soft_clustering_b_0':{'label':'MBC Binomial (Non-relevant)', 'color':_G_COLORS_0[2]}, 'pbm_soft_clustering_g_0':{'label':'MBC Gaussian (Non-relevant)', 'color':_G_COLORS_0[0]}}
  
  mpl.rcParams['pdf.fonttype'] = 42
  mpl.rc('font',family='Times New Roman')

  true_clicks_list = get_position_separated_ctr_lists(corrections[correction].clicks['train'], corrections[correction].doclist_ranges['train'], topk, True)
  xlim_max = 1.
  for pos in pos_list:
    rel = rel_b[:,pos]
    rel_ind = rel == 1
    nrel_ind = rel == 0
    clicks = true_clicks_list[pos]
    ctr = clicks[:,0] / clicks[:,1]
    
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(-0.1,2.1)
#     plt.xlim(0.,xlim_max)
#     xlim_max /= 2.
    datapoints = {'plot_type':'scatter', 'xlogscale':False}
    for correction in corrections:
      crctd = correcteds[correction][:,pos]
#       print(crctd.shape())
      x,y = get_unique_xy(ctr[rel_ind], crctd[rel_ind])
      plt.scatter(x, y+0.05, **legend_info[correction + '_1'])
      datapoints[correction + '_1'] = {'x':x, 'y':y+0.05, 'legend_info':legend_info[correction + '_1']}
      x,y = get_unique_xy(ctr[nrel_ind], crctd[nrel_ind])
      plt.scatter(x, y, **legend_info[correction + '_0'])
      datapoints[correction + '_0'] = {'x':x, 'y':y, 'legend_info':legend_info[correction + '_0']}
#     plt.legend(fontsize=22.2, facecolor='white', framealpha=1)
#     plt.show()
    plt.savefig('{}_pos{}.pdf'.format(prefix, pos),bbox_inches='tight', pad_inches=0)
    plt.clf()
    datapoints['legends_order'] = ['pbm_soft_clustering_1', 'pbm_soft_clustering_0', 'pbm_affine_1', 'pbm_affine_0']
    with open('{}_pos{}.pdf.pkl'.format(prefix, pos), 'wb') as f:
      pickle.dump(datapoints, f, protocol=2)
    
#   figlegend = plt.figure()
# #   figlegend.legend(handles=[mlines.Line2D([], [], linewidth=8, **legend_info[l]) for l in ['pbm_soft_clustering_1', 'pbm_soft_clustering_0', 'pbm_affine_1', 'pbm_affine_0']],
#   figlegend.legend(handles=[mlines.Line2D([], [], linewidth=8, **legend_info[l]) for l in ['pbm_soft_clustering_g_1', 'pbm_soft_clustering_g_0', 'pbm_soft_clustering_b_1', 'pbm_soft_clustering_b_0']],
#                    fontsize=18,
#                    loc='center',
#                    ncol=4,
#                    frameon=False,
#                    borderaxespad=0,
#                    borderpad=0,
#                    labelspacing=0.2,
#                    columnspacing=3.)
# #   figlegend.savefig(os.path.join(os.path.dirname(prefix),'insider_look_legend.pdf'),
#   figlegend.savefig(os.path.join(os.path.dirname(prefix),'mixture_insider_look_legend.pdf'),
#                     bbox_inches='tight')#, pad_inches=0)
  
  
def save_legend():
  _G_COLORS_0 = ['#f4a582', '#b8e186', '#92c5de', '#b2abd2']
  _G_COLORS = ['#ca0020', '#4dac26', '#0571b0', '#5e3c99']
  legend_info = {'pbm_affine_1':{'label':'AC (Relevant)', 'color':_G_COLORS[1]}, 'pbm_soft_clustering_1':{'label':'MBC (Relevant)', 'color':_G_COLORS[0]},
                 'pbm_affine_0':{'label':'AC (Non-relevant)', 'color':_G_COLORS_0[1]}, 'pbm_soft_clustering_0':{'label':'MBC (Non-relevant)', 'color':_G_COLORS_0[0]},
                 'pbm_soft_clustering_b_1':{'label':'MBC Binomial (Relevant)', 'color':_G_COLORS[2]}, 'pbm_soft_clustering_g_1':{'label':'MBC Gaussian (Relevant)', 'color':_G_COLORS[0]},
                 'pbm_soft_clustering_b_0':{'label':'MBC Binomial (Non-relevant)', 'color':_G_COLORS_0[2]}, 'pbm_soft_clustering_g_0':{'label':'MBC Gaussian (Non-relevant)', 'color':_G_COLORS_0[0]}}
  
  
  figlegend = plt.figure()
  figlegend.legend(handles=[mlines.Line2D([], [], linewidth=8, **legend_info[l]) for l in ['pbm_soft_clustering_1', 'pbm_soft_clustering_0', 'pbm_affine_1', 'pbm_affine_0']],
                   fontsize=18,
                   loc='center',
                   ncol=4,
                   frameon=False,
                   borderaxespad=0,
                   borderpad=0,
                   labelspacing=0.2,
                   columnspacing=3.)
#   figlegend.savefig(os.path.join('/Users/aliv/Dropbox/MyPapers/2020-clustering-cltr/sections/figure','insider_look_legend.pdf'),
  figlegend.savefig(os.path.join('/Users/aliv/Dropbox/MyPapers/2020-clustering-cltr/sections/figure','mixture_insider_look_legend.pdf'),
                    bbox_inches='tight')#, pad_inches=0)
  
  
  
  
  
  