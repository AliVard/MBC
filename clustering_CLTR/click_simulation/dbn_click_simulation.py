'''
Created on 4 Jun 2020

@author: aliv
'''

import numpy as np

from .base import BaseClickSimulation, _check_type, clean_and_sort, compute_attraction_probs
from ..mLogger import mLoggers

def _dbn_clicks_perquery(attraction_prob, rel_prob, gamma):
  s_u = rel_prob / attraction_prob
  s_u[rel_prob == 0] = 0.
  s_u[s_u > 1] = 1.
  clicks = []
  for pos in range(attraction_prob.shape[0]):
    c = np.random.binomial(1, attraction_prob[pos])
    s = 0.
    if c == 1:
      s = np.random.binomial(1, s_u[pos])
      clicks.append(pos)
      if s == 1 or np.random.binomial(1, gamma) == 0:
        break
  return np.array(clicks, dtype=np.int16)

class DBNClickSimulation(BaseClickSimulation):
  def __init__(self, model_name, json_description, default_topk=10000):
    super(DBNClickSimulation, self).__init__(model_name, json_description, default_topk)
    _check_type(self)
    
  policy_type = 'dbn'
  json_gamma_key = 'gamma'
  json_epsilonplus_key = 'epsilon+'
  json_epsilonminus_key = 'epsilon-'
  
  def _init_parameters(self):
    level2prob = lambda x: float(x)
    if self.json_level2prob_key in self.json_description:
      level2prob = eval(self.json_description[self.json_level2prob_key])
    topk = self.default_topk
    if self.json_topk_key in self.json_description:
      topk = self.json_description[self.json_topk_key]
    
    self.topk = topk
    self.level2prob = level2prob
    
    if self.json_gamma_key in self.json_description:
      gamma = eval(self.json_description[self.json_gamma_key])
      epsilon_p = eval(self.json_description[self.json_epsilonplus_key])
      epsilon_n = eval(self.json_description[self.json_epsilonminus_key])
      
        
      _epsilon_p = np.zeros(self.topk) + epsilon_p[-1]
      _epsilon_n = np.zeros(self.topk) + epsilon_n[-1]
      _epsilon_p[:min(self.topk,len(epsilon_p))] = epsilon_p[:min(self.topk,len(epsilon_p))]
      _epsilon_n[:min(self.topk,len(epsilon_n))] = epsilon_n[:min(self.topk,len(epsilon_n))]
          
      
      self.gamma = gamma
      self.epsilon_p = _epsilon_p
      self.epsilon_n = _epsilon_n
    else:
      raise ValueError('no suitable parameters for initializing {} parameters'.format(self.__class__.__name__))

    
    self.is_parameters_init = True
    
  def simulate_clicks(self, data_fold_split, ranks, click_count, output_pickle_path):
    if not self.is_parameters_init:
      self._init_parameters()
      
    argsorted, doclist_ranges = clean_and_sort(ranks, data_fold_split, self.topk)
    labels = data_fold_split.label_vector[argsorted]
    rel_probs = np.array(list(map(self.level2prob, labels)))
    attraction_probs = compute_attraction_probs(doclist_ranges, rel_probs, self.epsilon_p - self.epsilon_n, self.epsilon_n)
    
    clicks = [[] for _ in range(doclist_ranges.shape[0] - 1)]
    cnt = 0
    while cnt < click_count:
      id = np.random.choice(doclist_ranges.shape[0] - 1)
      positions = _dbn_clicks_perquery(attraction_probs[doclist_ranges[id]:doclist_ranges[id+1]], rel_probs[doclist_ranges[id]:doclist_ranges[id+1]], self.gamma)
      
      clicks[id].append(positions)
      cnt += positions.shape[0]
        
    self.save_pickle(clicks, argsorted, doclist_ranges, output_pickle_path)
    
    