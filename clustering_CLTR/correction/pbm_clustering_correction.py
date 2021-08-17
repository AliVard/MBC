'''
Created on 5 Jun 2020

@author: aliv
'''
from .base import BaseCorrection, positions2ctr
from ..gaussian_mixture import GaussianMixture
from ..binomial_mixture import BinomialMixture

from ..click_simulation.pbm_click_simulation import PBMClickSimulation

import numpy as np

from ..mLogger import mLoggers

from scipy.special import binom
# logger = logging.getLogger('clustering_correction')


def binom_prob_eq(p, n, k):
  return binom(n, k) * (p**k) * ((1.-p)**(n-k))

def get_position_separated_ctr_lists(clicks, doclist_ranges, topk, padd_zero = False):
  """Returns a list of nparrays, one array of ctr per position
  Parameters
  ----------
  clicks: the first output of click_simulation.base.load_clicks_pickle
          i.e. a list of queries -> clicks[qid] contains a list of sessions for query 'qid' -> clicks[qid][sid] is an np array containing the positions of clicked items at session 'sid'
  doclist_ranges:  the third output of click_simulation.base.load_clicks_pickle
  topk:  max rank (position) of all queries
  padd_zero:  if 'True' the ctr of all queries is padded to have 'topk' values
  
  Returns
  -------
  ctr_lists:  A list of size 'topk' containing numpy arrays of ctr for each position.
              ctr_lists[position] is a (Q, 2) shape array, 'Q' being the number of queries which have at least 'position' number of items.
              the first coulmn of ctr_lists[position] is the number of clicks and the second column is the number of impressions
  """
  ctr_lists = [[] for _ in range(topk)]
  
  for id in range(doclist_ranges.shape[0] - 1):
    max_rank = doclist_ranges[id+1] - doclist_ranges[id]
    ctrs, n = positions2ctr(clicks[id], max_rank)
#     if n > 0:
    if True:
      for position in range(max_rank):
        ctr_lists[position].append([[ctrs[position], n]])
      if padd_zero:
        for position in range(max_rank, topk):
          ctr_lists[position].append([[0., n]])
      
  return [np.concatenate(ctr_lists[i], 0) for i in range(topk)]

def rate_to_labels(MM, rate):
  y = MM.predict(rate)
  if MM.means_.shape[1] == 2:
    argsorted = np.argsort(MM.means_[:,0]/MM.means_[:,1])
  else:
    argsorted = np.argsort(MM.means_[:,0])
  true_y = np.zeros_like(y)
  for i in range(MM.n_components):
    true_y[y==argsorted[i]]=i
    
#   print(list(rate[:40,0]))
#   print(list(true_y[:40]))
  return true_y

def rate_to_soft_labels(BMM, rate):
#   if BMM.means_.shape[1] == 1:
#     raise ValueError('soft labels only work for BMM')
  
  if BMM.means_.shape[0] == 1:
    return np.zeros_like(rate[:,1])
  y = BMM.soft_predict(rate)
  if BMM.means_.shape[1] == 1:
    means_ = BMM.means_[:,0]
  else:
    means_ = BMM.means_[:,0]/BMM.means_[:,1]
  argsorted = np.argsort(means_)
  
  true_y = y[:,argsorted[-1]]

#   true_y[true_y>0.9] = 1
#   true_y[true_y<0.1] = 0
  
  true_y[rate[:,0]==0] = 0
    
  return true_y

class PBMClusteringCorrection(BaseCorrection):
  def __init__(self, **kwargs):
    super(PBMClusteringCorrection, self).__init__(**kwargs)
    
  def _init_parameters(self):
    if 'mixture' in self.params_kwargs:
      self.mixture = self.params_kwargs['mixture']
      self.params_kwargs.pop('mixture')
    else:
      self.mixture = 'binomial'
      
    self._soft = self.params_kwargs['soft']
    self.params_kwargs.pop('soft')
    
    self._enhance_mult = self.params_kwargs['enhance_mult']
    self.params_kwargs.pop('enhance_mult')
    self._enhance_dist = self.params_kwargs['enhance_dist']
    self.params_kwargs.pop('enhance_dist')
    
    weights = None
    if 'oracle' in self.params_kwargs:
      weights = self.params_kwargs['oracle']['weights']
      clickSimulation = PBMClickSimulation('', self.params_kwargs['oracle']['json'], self.topk)
      clickSimulation._init_parameters()
      alpha = clickSimulation.alpha
      beta = clickSimulation.beta
      self.params_kwargs.pop('oracle')
      
      
    model_class = {'binomial':BinomialMixture, 
                   'gaussian':GaussianMixture}[self.mixture]
                  
    if self.params_kwargs['n_components'] == 0:
      self.params_kwargs['n_components'] = 1
      self.models_1_component = [model_class(**self.params_kwargs) for _ in range(self.topk)]
      self.params_kwargs['n_components'] = 2
      self.models_2_component = [model_class(**self.params_kwargs) for _ in range(self.topk)]
      self.params_kwargs['n_components'] = 3
      self.models_3_component = [model_class(**self.params_kwargs) for _ in range(self.topk)]
      self.models = []
    else:
      self.models = [model_class(**self.params_kwargs) for _ in range(self.topk)]
      

    self.bics = []     
    if weights is None:
#       print(self.bics)
      self._fit_train_clicks('train')
    else:
      for i in range(self.topk):
        self.models[i].weights_ = np.array(weights[i])
        self.models[i].means_ = np.ones([2,2])
        self.models[i].means_[0,0] = beta[i] 
        self.models[i].means_[1,0] = beta[i] + alpha[i]
        
    for i in range(self.topk):
      if self.models[i].means_.shape[1] == 2:
        means_ = self.models[i].means_[:,0]/self.models[i].means_[:,1]
      else:
        means_ = self.models[i].means_[:,0]
      mLoggers['clustering_correction'].debug('model_{} weights:{}, bics:{}; mean:{}'.
#             format(i,self.models[i].weights_, self.models[i].means_[:,0]/self.models[i].means_[:,1]), self.models[i].bic(),
            format(i,self.models[i].weights_, self.bics[i],
            means_))



    self.is_parameters_init = True
    
    
  def _enhance_train_data(self, ctr_list):
    if self._enhance_dist == 'uniform':
      for i in range(ctr_list.shape[0]):
        if ctr_list[i,0] < 20:
          if ctr_list[i,0] < ctr_list[i,1]:
            ctr_list[i,0] = ctr_list[i,0]*self._enhance_mult + np.random.randint(self._enhance_mult)
            ctr_list[i,1] = ctr_list[i,1]*self._enhance_mult
    if self._enhance_dist == 'binomial':
      for i in range(ctr_list.shape[0]):
        if ctr_list[i,0] < 20:
          if ctr_list[i,0] < ctr_list[i,1]:
            p = (ctr_list[i,0] + 0.5) / ctr_list[i,1]
            n = ctr_list[i,1]*self._enhance_mult
            probs = []
            sum = 0
            for k in range(int(ctr_list[i,0]*self._enhance_mult), int((ctr_list[i,0]+1)*self._enhance_mult)):
              probs.append(binom_prob_eq(p,n,k))
              sum += probs[-1]
            for k in range(len(probs)):
              probs[k] /= sum
            ctr_list[i,0] = int(ctr_list[i,0]*self._enhance_mult) + np.random.choice(len(probs),1,probs)[0]
            ctr_list[i,1] = n
          
      
    
  def _enhance_train_data_attempt(self, ctr_list):
    for i in range(ctr_list.shape[0]):
      if ctr_list[i,0] < ctr_list[i,1]:
        ctr_list[i,0] = ctr_list[i,0] + np.random.randint(self._enhance) * 1. / self._enhance
      
    
  def _fit_train_clicks(self, train_key):
    clicks = self.clicks[train_key]
    doclist_ranges = self.doclist_ranges[train_key]
    ctr_lists = get_position_separated_ctr_lists(clicks, doclist_ranges, self.topk)
#     ctr_lists_ = ctr_lists[ctr_lists[:,1]>0,:]
#     ctr_lists = ctr_lists_
    
    auto_n_components = len(self.models) == 0
      
    for i in range(self.topk):
      mLoggers['enhance'].debug('original ctr: {}'.format(ctr_lists[i][:5,:]))
      self._enhance_train_data(ctr_lists[i])
      mLoggers['enhance'].debug('enhanced ctr: {}'.format(ctr_lists[i][:5,:]))
      non_nan_ind = ctr_lists[i][:,1]>0
      if self.mixture == 'gaussian':
        normalized = ctr_lists[i][non_nan_ind,0:1] / ctr_lists[i][non_nan_ind,1:2]
        ctr_lists[i] = normalized
      else:
        removed_nan = ctr_lists[i][non_nan_ind,:]
        ctr_lists[i] = removed_nan
      if auto_n_components:
        self.models_1_component[i].fit(ctr_lists[i])
        self.models_2_component[i].fit(ctr_lists[i])
        self.models_3_component[i].fit(ctr_lists[i])
        self.bics.append((self.models_1_component[i].bic(ctr_lists[i]), self.models_2_component[i].bic(ctr_lists[i]), self.models_3_component[i].bic(ctr_lists[i])))
        min_bic = min(self.bics[-1])
        if min_bic == self.bics[-1][0]:
          self.models.append(self.models_1_component[i])
        elif min_bic == self.bics[-1][1]:
          self.models.append(self.models_2_component[i])
        else:
          self.models.append(self.models_3_component[i])
      else:
        self.models[i].fit(ctr_lists[i])
        self.bics.append(self.models[i].bic(ctr_lists[i]))
                          
      GMM = self.models[i]
      if GMM.means_.shape[1] == 2:
        argsorted = np.argsort(GMM.means_[:,0]/GMM.means_[:,1])
      else:
        argsorted = np.argsort(GMM.means_[:,0])
#       print('position {}, means: {}'.format(i,GMM.means_[argsorted,0]/GMM.means_[argsorted,1]))
  
  
  def _correct_clicks(self, clicks, doclist_ranges):
    corrected_clicks = []
    ctr_lists = get_position_separated_ctr_lists(clicks, doclist_ranges, self.topk, True)
    
    if self.mixture == 'gaussian':
      for i in range(self.topk):
        normalized = ctr_lists[i][:,0:1] / ctr_lists[i][:,1:2]
        normalized[ctr_lists[i][:,1]==0] = 0.
        ctr_lists[i] = normalized
        
        
    for i in range(self.topk):
      if self._soft == 'soft':
        corrected_clicks.append(rate_to_soft_labels(self.models[i],ctr_lists[i])[:,np.newaxis])
      else:
        corrected_clicks.append(rate_to_labels(self.models[i],ctr_lists[i])[:,np.newaxis])

      
    corrected_clicks = np.concatenate(corrected_clicks, 1)
    
#     print(np.mean(corrected_clicks[:,:12],0))
    
    removed_pads = np.zeros([doclist_ranges[-1]], dtype=np.float64)
    for id in range(doclist_ranges.shape[0] - 1):
#       print('{}, {}:{}'.format(id, doclist_ranges[id],doclist_ranges[id+1]))
      removed_pads[doclist_ranges[id]:doclist_ranges[id+1]] = corrected_clicks[id,:(doclist_ranges[id+1] - doclist_ranges[id])]
      
    return removed_pads