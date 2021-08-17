'''
Created on 9 Jun 2020

@author: aliv
'''

from ..metrics import LTRMetrics
from ..correction.base import BaseCorrection
# from abc import ABCMeta, abstractmethod
import os
import numpy as np
import pickle
import time
from ..mLogger import mLoggers

def _assign_combined_fields(datafold_split, correction, split_name):
  if split_name not in correction.corrected_clicks:
    return None
  combined_split = type('', (), {})()
  combined_split.feature_matrix = datafold_split.feature_matrix[correction.argsorted[split_name]]
  combined_split.doclist_ranges = correction.doclist_ranges[split_name]
  combined_split.label_vector = correction.corrected_clicks[split_name]
  return combined_split
  
def combine_corrected_clicks_and_features(train_clicks_pickle_path, datafold, correction = None):
  """combines the clicks with the features to be used in the learner
  Parameters
  ----------
  train_clicks_pickle_path : full path to the click pickle file of the 'train' set. Path to 'valid' and 'test' sets are built from this address.
  datafold : the datafold consisting of three splits: 'train', 'valid' and 'test'
  correction : An instance from a correction class, implementing 'BaseCorrection'. If 'None' it is instantiated with 'BaseCorrection'.
  
  Returns
  -------
  combined :  an object consisting of three splits: 'train' and 'valid'. Each split has 'feature_matrix', 'doclist_ranges' and 'label_vector'.
              The 'label_vector' is the corrected clicks obtained from 'correction'.
              For 'test', the 'label_vector' is the true relevance values stored in the 'datafold.test'.
  """
  train_dir = os.path.dirname(train_clicks_pickle_path)
  train_file = os.path.basename(train_clicks_pickle_path)
  valid_file = train_file.replace('train','valid')
#   test_file = train_file.replace('train','test')
  
  if correction is None:
    correction = BaseCorrection()
    
  correction.add_clicks_pickle_path('train', os.path.join(train_dir,train_file))
  if os.path.exists(os.path.join(train_dir, valid_file)):
    correction.add_clicks_pickle_path('valid', os.path.join(train_dir,valid_file))
#   if os.path.exists(os.path.join(train_dir, test_file)):
#     correction.add_clicks_pickle_path('test', os.path.join(train_dir,test_file))
    
  start_correction_time = time.time()
  correction.correct()
  mLoggers['time'].debug('correction took: {} secs.'.format(time.time() - start_correction_time))
  
  combined = type('', (), {})()
  combined.train = _assign_combined_fields(datafold.train, correction, 'train')
  combined.valid = _assign_combined_fields(datafold.valid, correction, 'valid')
#   combined.test = _assign_combined_fields(datafold.test, correction, 'test')
#   combined.test.label_vector = datafold.test.label_vector[correction.argsorted['test']]
  return combined

def ndcg_binary_labels(y_true, ranges, y_pred, k):
  lv = np.zeros_like(y_true)
  lv[y_true>2] = 1
  metric = LTRMetrics(lv,np.diff(ranges),y_pred)
  return metric.NDCG(k), metric.MAP()


def ndcg_graded_labels(y_true, ranges, y_pred, k):
  metric = LTRMetrics(y_true,np.diff(ranges),y_pred)
  return metric.NDCG(k), metric.MAP()


def dcg_binary_labels(y_true, ranges, y_pred, k):
  lv = np.zeros_like(y_true)
  lv[y_true>2] = 1
  metric = LTRMetrics(lv,np.diff(ranges),y_pred)
  return metric.DCG(k)

class BaseLearner(object):
  def __init__(self):
    pass
  
#   @abstractmethod
  def load_saved_model(self, path):
    pass
    
#   @abstractmethod
  def train(self, trainset, validset):
    pass
  
#   @abstractmethod
  def predict(self, testset):
    pass
  

  def test(self, testset, metric = None, save_to = None, binary_rel = True):
    if metric is None:
      if binary_rel:
        metric = ndcg_binary_labels
      else:
        metric = ndcg_graded_labels
    y_pred = self.predict(testset)
    if save_to is not None:
      results = {}
      results['doclist_ranges'] = testset.doclist_ranges
      results['label_vector'] = testset.label_vector
      results['predictions'] = y_pred
      with open(save_to, 'wb') as f:
        pickle.dump(results, 
                    f, 
                    protocol=2)
    return metric(testset.label_vector, testset.doclist_ranges, y_pred, 10)
# 
#   def test(self, testset, metric = ndcg_binary_labels):
#     y_pred = self.predict(testset)
#     lv = testset.label_vector
#     lv = np.zeros_like(testset.label_vector)
#     lv[testset.label_vector>2] = 1
#     metric = LTRMetrics(lv,np.diff(testset.doclist_ranges),y_pred)
#     return metric.NDCG(10)