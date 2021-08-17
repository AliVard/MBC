'''
Created on 4 Jun 2020

@author: aliv
'''

import numpy as np

from .base import BaseClickSimulation, _check_type, clean_and_sort
from ..mLogger import mLoggers


def _compute_click_probs(doclist_ranges, rel_probs, alpha, beta):
  click_probs = []
  mLoggers['simulation'].debug(beta)
  mLoggers['simulation'].debug(alpha + beta)
  for qid in range(doclist_ranges.shape[0] - 1):
    slice = rel_probs[doclist_ranges[qid]:doclist_ranges[qid+1]]
    click_probs.append(alpha[:slice.shape[0]] * slice + beta[:slice.shape[0]])
    if qid < 400:
      mLoggers['simulation'].debug('pos{}: \nrel:{}\nprobs:{}'.format(qid,slice,click_probs[-1]))
  
  return np.concatenate(click_probs, 0)
  
class PBMClickSimulation(BaseClickSimulation):
  def __init__(self, model_name, json_description, default_topk=10000):
    super(PBMClickSimulation, self).__init__(model_name, json_description, default_topk)
    _check_type(self)
    
  policy_type = 'pbm'
  json_theta_key = 'theta'
  json_epsilonplus_key = 'epsilon+'
  json_epsilonminus_key = 'epsilon-'
  json_zetaplus_key = 'zeta+'
  json_zetaminus_key = 'zeta-'
  json_alpha_key = 'alpha'
  json_beta_key = 'beta'
  
  def _init_parameters(self):
    level2prob = lambda x: float(x)
    if self.json_level2prob_key in self.json_description:
      level2prob = eval(self.json_description[self.json_level2prob_key])
    topk = self.default_topk
    if self.json_topk_key in self.json_description:
      topk = self.json_description[self.json_topk_key]
    
    self.topk = topk
    self.level2prob = level2prob
    
    if self.json_alpha_key in self.json_description:
      alpha = eval(self.json_description[self.json_alpha_key])
      beta = eval(self.json_description[self.json_beta_key])
        
      
      _alpha = np.zeros(topk) + alpha[-1]
      _beta = np.zeros(topk) + beta[-1]
      _alpha[:len(alpha)] = alpha
      _beta[:len(beta)] = beta
      
      self.alpha = _alpha
      self.beta = _beta
    elif self.json_theta_key in self.json_description:
      theta = eval(self.json_description[self.json_theta_key])
      epsilon_p = eval(self.json_description[self.json_epsilonplus_key])
      epsilon_n = eval(self.json_description[self.json_epsilonminus_key])
        
      
      _theta = np.zeros(topk) + theta[-1]
      _epsilon_p = np.zeros(topk) + epsilon_p[-1]
      _epsilon_n = np.zeros(topk) + epsilon_n[-1]
      _theta[:len(theta)] = theta
      _epsilon_p[:len(epsilon_p)] = epsilon_p
      _epsilon_n[:len(epsilon_n)] = epsilon_n
      
      self.alpha = _theta * (_epsilon_p - _epsilon_n)
      self.beta = _theta * _epsilon_n
    elif self.json_zetaplus_key in self.json_description:
      zeta_p = eval(self.json_description[self.json_zetaplus_key])
      zeta_n = eval(self.json_description[self.json_zetaminus_key])
        
      
      _zeta_p = np.zeros(topk) + zeta_p[-1]
      _zeta_n = np.zeros(topk) + zeta_n[-1]
      _zeta_p[:len(zeta_p)] = zeta_p
      _zeta_n[:len(zeta_n)] = zeta_n
      
      self.alpha = _zeta_p - _zeta_n
      self.beta = _zeta_n
    else:
      raise ValueError('no suitable parameters for initializing {} parameters'.format(self.__class__.__name__))

    
    self.is_parameters_init = True
    
  def simulate_clicks(self, data_fold_split, ranks, click_count, output_pickle_path):
    if not self.is_parameters_init:
      self._init_parameters()
      
    argsorted, doclist_ranges = clean_and_sort(ranks, data_fold_split, self.topk)
    labels = data_fold_split.label_vector[argsorted]
    rel_probs = np.array(list(map(self.level2prob, labels)))
    click_probs = _compute_click_probs(doclist_ranges, rel_probs, self.alpha, self.beta)
    clicks = [[] for _ in range(doclist_ranges.shape[0] - 1)]
    cnt = 0
    while cnt < click_count:
      id = np.random.choice(doclist_ranges.shape[0] - 1)
      clicked = np.random.binomial(1, click_probs[doclist_ranges[id]:doclist_ranges[id+1]])
      if id < 20:
        mLoggers['simulation'].debug('pos{}: clicked:{}'.format(id, clicked))
      positions = np.where(clicked==1)[0].astype(np.int16)
#       if positions.shape[0] > 0:
      clicks[id].append(positions)
      cnt += positions.shape[0]
        
    self.save_pickle(clicks, argsorted, doclist_ranges, output_pickle_path)
    
    