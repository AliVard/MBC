'''
Created on 4 Jun 2020

@author: aliv
'''

# from abc import ABCMeta, abstractmethod
import pickle
import numpy as np

def _check_type(self):
  """checks the type of json description matches the class or not """
  assert self.json_description[self.json_type_key]==self.policy_type, '{} can only be built from a "{}":"{}" policy'.format(self.__class__.__name__, self.json_type_key, self.policy_type)
    
def load_clicks_pickle(pickle_path):
  """Used to load click pickles that are saved by descendants of this class
  Parameters
  ----------
  pickle_path : full path to the click pickle file

  Returns
  -------
  clicks: a list of queries -> clicks[qid] contains a list of sessions for query 'qid' -> clicks[qid][sid] is an np array containing the positions of clicked items at session 'sid'
  argsorted: the inverse ranking of items shown to the user. feature_matrix[argsorted] would be aligned to the 'clicks'
  doclist_ranges: the new ranges of doclist, consistent with argsorted
  
  See Also
  --------
  clean_and_sort
  """
  with open(pickle_path, 'rb') as f:
    clicks_pickle = pickle.load(f, encoding='latin1')
    
  return clicks_pickle['clicks'],clicks_pickle['argsorted'],clicks_pickle['doclist_ranges']

def clean_and_sort(ranks, data_fold_split, topk):
  """sorts the items based on the 'ranks' input and removes ranks after 'topk'
  Returns
  -------
  argsorted: the inverse ranking of items shown to the user. feature_matrix[argsorted] would be aligned to the 'clicks'
  doclist_ranges: the new ranges of doclist, consistent with argsorted
  
  See Also
  --------
  load_clicks_pickle"""
  argsorted = []
  new_doclist_range = [0]
  
  for qid in range(data_fold_split.doclist_ranges.shape[0] - 1):
    irank = np.argsort(ranks[data_fold_split.doclist_ranges[qid]:data_fold_split.doclist_ranges[qid+1]])
    shown_len = min(irank.shape[0], topk)
    argsorted.append(data_fold_split.doclist_ranges[qid] + irank[:shown_len])
    new_doclist_range.append(shown_len)
    
  _argsorted = np.concatenate(argsorted, axis=0)
  _doclist_range = np.cumsum(np.array(new_doclist_range), axis=0)
  
  return _argsorted, _doclist_range
    

def compute_attraction_probs(doclist_ranges, rel_probs, alpha, beta):
  attraction_probs = []
  for qid in range(doclist_ranges.shape[0] - 1):
    slice = rel_probs[doclist_ranges[qid]:doclist_ranges[qid+1]]
    attraction_probs.append(alpha[:slice.shape[0]] * slice + beta[:slice.shape[0]])
  
  return np.concatenate(attraction_probs, 0)

class BaseClickSimulation(object):
  """Base class for click simulation.
  It saves the clicks in a pickle as described in 'load_clicks_pickle'"""
  def __init__(self, model_name, json_description, default_topk):
    assert BaseClickSimulation.json_type_key in json_description, 'click policies should have "{}"'.format(BaseClickSimulation.json_type_key)
    
    self.model_name = model_name
    self.json_description = json_description
    self.default_topk = default_topk
    
    self.is_parameters_init = False
    
  json_type_key = 'type'
  json_topk_key = 'topk'
  json_level2prob_key = 'level2prob'
  pickle_protocol_number = 2
  
#   @abstractmethod
  def _init_parameters(self):
      pass
    
#   @abstractmethod
  def simulate_clicks(self, data_fold_split, ranks, click_count, output_pickle_path):
      pass
    
  def save_pickle(self, clicks, argsorted, doclist_ranges, pickle_path):
    with open(pickle_path, 'wb') as f:
      pickle.dump({'clicks':clicks, 'argsorted':argsorted, 'doclist_ranges':doclist_ranges}, 
                  f, 
                  protocol=self.pickle_protocol_number)
    
    