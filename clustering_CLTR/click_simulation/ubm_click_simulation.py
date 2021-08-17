'''
Created on 4 Jun 2020

@author: aliv
'''

import numpy as np

from .base import BaseClickSimulation, _check_type, clean_and_sort, compute_attraction_probs
from ..mLogger import mLoggers

def _ubm_clicks_perquery(attraction_prob, exam):
  clicks = []
  last_click = exam.shape[0] - 1
  for pos in range(attraction_prob.shape[0]):
    c = np.random.binomial(1, exam[pos,last_click] * attraction_prob[pos])
    if c == 1:
      clicks.append(pos)
      last_click = pos
  return np.array(clicks, dtype=np.int16)

class UBMClickSimulation(BaseClickSimulation):
  def __init__(self, model_name, json_description, default_topk=10000):
    super(UBMClickSimulation, self).__init__(model_name, json_description, default_topk)
    _check_type(self)
    
  policy_type = 'ubm'
  json_exam_key = 'exam'
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
    
    if self.json_exam_key in self.json_description:
      exam = eval(self.json_description[self.json_exam_key])
      epsilon_p = eval(self.json_description[self.json_epsilonplus_key])
      epsilon_n = eval(self.json_description[self.json_epsilonminus_key])
      
      if self.topk > len(exam):
        self.topk = len(exam)
        
      _epsilon_p = np.zeros(self.topk) + epsilon_p[-1]
      _epsilon_n = np.zeros(self.topk) + epsilon_n[-1]
      _epsilon_p[:min(self.topk,len(epsilon_p))] = epsilon_p[:min(self.topk,len(epsilon_p))]
      _epsilon_n[:min(self.topk,len(epsilon_n))] = epsilon_n[:min(self.topk,len(epsilon_n))]
          
      
      self.exam = np.array(exam)
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
      positions = _ubm_clicks_perquery(attraction_probs[doclist_ranges[id]:doclist_ranges[id+1]], self.exam)
      
      clicks[id].append(positions)
      cnt += positions.shape[0]
        
    self.save_pickle(clicks, argsorted, doclist_ranges, output_pickle_path)
    
    