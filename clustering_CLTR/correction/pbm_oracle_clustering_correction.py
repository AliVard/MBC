'''
Created on 29 Jun 2020

@author: aliv
'''
from .base import BaseCorrection, positions2ctr
from ..click_simulation.pbm_click_simulation import PBMClickSimulation
from .pbm_clustering_correction import PBMClusteringCorrection, get_position_separated_ctr_lists

from ..binomial_mixture import OracleBinomialMixture, BinomialMixture

import numpy as np
import logging

logger = logging.getLogger('clustering_correction')


class PBMOracleClusteringCorrection(PBMClusteringCorrection):
  def __init__(self, **kwargs):
    super(PBMOracleClusteringCorrection, self).__init__(**kwargs)
    
    
  def _init_parameters(self):
    if self.params_kwargs is not None and 'json' in self.params_kwargs:
      clickSimulation = PBMClickSimulation('', self.params_kwargs['json'], self.topk)
      clickSimulation._init_parameters()
      self.alpha = clickSimulation.alpha
      self.beta = clickSimulation.beta
    else:
      self.alpha = np.ones([self.topk], dtype=np.float64) - 1.e-3
      self.beta = np.zeros([self.topk], dtype=np.float64) + 1.e-6
    
    self.params_kwargs.pop('json')
    
    self._soft = self.params_kwargs['soft']
    self.params_kwargs.pop('soft')
      
    self.mixture = 'binomial'
    logger.debug('beta:{},\nalpha:{}'.format(self.beta,self.alpha))
    self.models = [BinomialMixture(**self.params_kwargs) for i in range(self.topk)]
#     self.models = [OracleBinomialMixture(means=np.array([self.beta[i], self.beta[i]+self.alpha[i]]),**self.params_kwargs) for i in range(self.topk)]
                  
    self._fit_train_clicks('train')
    for i in range(self.topk):
      means=np.array([self.beta[i], self.beta[i]+self.alpha[i]])
      
      argsorted = np.argsort(self.models[i].means_[:,0]/self.models[i].means_[:,1])
      
      self.models[i].means_[argsorted,:] = np.concatenate([means[:,np.newaxis],np.ones_like(means[:,np.newaxis])],1)
      
    for i in range(self.topk):
      logger.debug('model_{} weights:{}'.format(i,self.models[i].weights_))
      logger.debug('model_{} means:{}'.format(i,self.models[i].means_[:,0]/self.models[i].means_[:,1]))
    
    self.is_parameters_init = True
    
#     
#   def _correct_clicks(self, clicks, doclist_ranges):
#     corrected_clicks = []
#     ctr_lists = get_position_separated_ctr_lists(clicks, doclist_ranges, self.topk, True)
#         
#     for i in range(self.topk):
#       rate = ctr_lists[i][:,0:1]/ctr_lists[i][:,1:2]
# #       rate *= (ctr_lists[i][:,1:2]-10)/ctr_lists[i][:,1:2]
#       labels = np.zeros_like(rate)
#       labels[rate>self.threshold[i]] = 1.
#       corrected_clicks.append(labels)
#       
#       
#     corrected_clicks = np.concatenate(corrected_clicks, 1)
#     print(np.mean(corrected_clicks[:,:12],0))
#     
#     removed_pads = np.zeros([doclist_ranges[-1]], dtype=np.float64)
#     for id in range(doclist_ranges.shape[0] - 1):
# #       print('{}, {}:{}'.format(id, doclist_ranges[id],doclist_ranges[id+1]))
#       removed_pads[doclist_ranges[id]:doclist_ranges[id+1]] = corrected_clicks[id,:(doclist_ranges[id+1] - doclist_ranges[id])]
#       
#     return removed_pads
#     