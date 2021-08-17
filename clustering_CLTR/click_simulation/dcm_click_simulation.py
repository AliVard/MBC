'''
Created on 4 Jun 2020

@author: aliv
'''

import numpy as np

from .base import BaseClickSimulation, _check_type, clean_and_sort, compute_attraction_probs
from ..mLogger import mLoggers

def _dcm_click_probs_perquery(attraction_prob, lambda_r):
  click_probs = np.zeros_like(attraction_prob)
  e = 1.
  for r in range(attraction_prob.shape[0]):
    click_probs[r] = e * attraction_prob[r]
    e = lambda_r[r] * click_probs[r] + (e - click_probs[r])
  return click_probs
  

def _dcm_click_probs(doclist_ranges, attraction_probs, lambda_r):
  click_probs = []
  for qid in range(doclist_ranges.shape[0] - 1):
    slice = attraction_probs[doclist_ranges[qid]:doclist_ranges[qid+1]]
    click_probs.append(_dcm_click_probs_perquery(slice, lambda_r))
  
  return np.concatenate(click_probs, 0)

class DCMClickSimulation(BaseClickSimulation):
  def __init__(self, model_name, json_description, default_topk=10000):
    super(DCMClickSimulation, self).__init__(model_name, json_description, default_topk)
    _check_type(self)
    
  policy_type = 'dcm'
  json_lambda_key = 'lambda'
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
    
    if self.json_lambda_key in self.json_description:
      lambda_r = eval(self.json_description[self.json_lambda_key])
      epsilon_p = eval(self.json_description[self.json_epsilonplus_key])
      epsilon_n = eval(self.json_description[self.json_epsilonminus_key])
          
        
      _lambda_r = np.zeros(topk) + lambda_r[-1]
      _epsilon_p = np.zeros(topk) + epsilon_p[-1]
      _epsilon_n = np.zeros(topk) + epsilon_n[-1]
      _lambda_r[:min(topk,len(lambda_r))] = lambda_r[:min(topk,len(lambda_r))]
      _epsilon_p[:min(topk,len(epsilon_p))] = epsilon_p[:min(topk,len(epsilon_p))]
      _epsilon_n[:min(topk,len(epsilon_n))] = epsilon_n[:min(topk,len(epsilon_n))]
          
      
      self.lambda_r = _lambda_r
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
    click_probs = _dcm_click_probs(doclist_ranges, attraction_probs, self.lambda_r)
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
    
    