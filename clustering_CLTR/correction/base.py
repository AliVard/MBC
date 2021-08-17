'''
Created on 5 Jun 2020

@author: aliv
'''

# from abc import ABCMeta, abstractmethod
import numpy as np

from ..click_simulation.base import load_clicks_pickle


def positions2ctr(positions_list,max_rank):
  ctr = np.zeros(max_rank, dtype=np.float64)
  for pos in positions_list:
    ctr[pos] += 1.
  return ctr, len(positions_list)
  
  
class BaseCorrection(object):
  def __init__(self, **kwargs):
    self.clicks, self.argsorted, self.doclist_ranges, self.corrected_clicks = {}, {}, {}, {}
    self.relevance_targets = {}
    self.topk = 0
    self.params_kwargs = kwargs
      
    self.is_parameters_init = False
    
  def add_clicks_pickle_path(self, name, clicks_pickle_path):
    clicks, argsorted, doclist_ranges = load_clicks_pickle(clicks_pickle_path)
    self.clicks[name] = clicks
    self.argsorted[name] = argsorted
    self.doclist_ranges[name] = doclist_ranges
    self.corrected_clicks[name] = None
    self.relevance_targets[name] = None
    
    max_rank = np.diff(doclist_ranges).max()
    if max_rank > self.topk:
      self.topk = max_rank
      
#   @abstractmethod
  def _init_parameters(self):
      self.is_parameters_init = True
    
#   @abstractmethod
  def _correct_clicks(self, clicks, doclist_ranges):
#     return clicks
    pass
  
  
  def correct(self):
    if not self.is_parameters_init:
      self._init_parameters()
    
    for name in self.clicks:
      self.corrected_clicks[name] = self._correct_clicks(clicks = self.clicks[name], 
                                                         doclist_ranges = self.doclist_ranges[name])

        