'''
Created on Fri Mar  6  2020

@author: aliv
'''

from __future__ import print_function


import numpy as np

import sys
import time
import os


SYS_PATH_APPEND_DEPTH = 2
SYS_PATH_APPEND = os.path.abspath(__file__)
for _ in range(SYS_PATH_APPEND_DEPTH):
  SYS_PATH_APPEND = os.path.dirname(SYS_PATH_APPEND)
sys.path.append(SYS_PATH_APPEND)

from clustering_CLTR.preprocess import simulate_clicks
from clustering_CLTR.preprocess import dataset


class ClickDataSplit:
  def __init__(self, data, clicks, split_name):
    self.name = split_name
    self.data_fold_split = data.data_fold_split_by_name[split_name]
    self.clicks_split = clicks.clicks_split_by_name[split_name]
    
    self.clicks_split.clean_and_sort()
    
    self.feature_matrix = self.data_fold_split.feature_matrix[self.clicks_split.argsorted]
    self.label_vector = self.data_fold_split.label_vector[self.clicks_split.argsorted]
    self.click_rates = self.clicks_split.click_rates
    self.click_counts = self.clicks_split.click_counts
    self.click_showns = self.clicks_split.click_showns
    self.doclist_ranges = self.clicks_split.new_doclist_range
    
#         (self.doclist_ranges[:100])
    self.max_ranklist_size = np.max(np.diff(self.doclist_ranges))
    
    self.last_used_batch_index = -1
    self.samples_size = self.doclist_ranges.shape[0] - 1
  
  
  def get_random_indexes(self, batch_size):
    if self.last_used_batch_index == -1:
      self.epochs = -1
      self.last_used_batch_index = self.samples_size
    
    if self.last_used_batch_index >= self.samples_size:
      self.permuted_indices = np.random.permutation(self.samples_size)
      self.last_used_batch_index = 0
      self.epochs += 1
      
    start = self.last_used_batch_index
    end = self.last_used_batch_index + batch_size
    if end > self.samples_size:
      end = self.samples_size
    self.last_used_batch_index = end
    
    return self.permuted_indices[start:end]
    
  
  def load_batch(self, qids):
    feature_matrix = []
    click_rates = []
    padded_labels = []
    padding_mask = []
    if qids is None:
      qids = range(self.samples_size)
    for qid in qids:
      s_i = self.doclist_ranges[qid]
      e_i = self.doclist_ranges[qid+1]
      feature_matrix.append(self.feature_matrix[s_i:e_i, :])
      feature_matrix.append(np.zeros([self.max_ranklist_size - e_i + s_i, self.feature_matrix.shape[1]], dtype=np.float64))
      click_rates.append(self.click_rates[s_i:e_i])
      click_rates.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      padded_labels.append(self.label_vector[s_i:e_i])
      padded_labels.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      padding_mask.append(np.ones([e_i - s_i], dtype=np.float64))
      padding_mask.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      
    self.padded_labels = np.concatenate(padded_labels, axis=0)
    return np.concatenate(feature_matrix, axis=0), np.concatenate(click_rates, axis=0), np.concatenate(padding_mask, axis=0)
    
  def load_batch_rel_clicks(self, qids):
    label_vector = []
    click_rates = []
    click_counts = []
    click_showns = []
    padded_labels = []
    padding_mask = []
    if qids is None:
      qids = range(self.samples_size)
    for qid in qids:
      s_i = self.doclist_ranges[qid]
      e_i = self.doclist_ranges[qid+1]
      label_vector.append(self.label_vector[s_i:e_i])
      label_vector.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      click_rates.append(self.click_rates[s_i:e_i])
      click_rates.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      click_counts.append(self.click_counts[s_i:e_i])
      click_counts.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      click_showns.append(self.click_showns[s_i:e_i])
      click_showns.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      padded_labels.append(self.label_vector[s_i:e_i])
      padded_labels.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      padding_mask.append(np.ones([e_i - s_i], dtype=np.float64))
      padding_mask.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      
    self.padded_labels = np.concatenate(padded_labels, axis=0)
    return np.concatenate(label_vector, axis=0), np.concatenate(click_rates, axis=0), np.concatenate(padding_mask, axis=0), np.concatenate(click_counts, axis=0), np.concatenate(click_showns, axis=0)
    
  def load_test_epoch(self):
    return self.data_fold_split.feature_matrix, self.data_fold_split.label_vector
  
  def test_stats(self):
    return '"queries": {}, "docs": {}'.format(self.data_fold_split.doclist_ranges.shape[0] - 1, self.data_fold_split.feature_matrix.shape[0])

  def test_labels(self):
    return self.data_fold_split.label_vector
  
  def test_doclist_ranges(self):
    return self.data_fold_split.doclist_ranges
  
class ClickData:
  def __init__(self, dataset_name, datasets_info_path, data_fold, clicks_path):
    data = dataset.get_dataset_from_json_info(
                      dataset_name,
                      datasets_info_path,
                    ).get_data_folds()[data_fold]
    data.read_data()
    
    if clicks_path is not None:
      clicks = simulate_clicks.load_clicks_pickle(clicks_path)
    else:
      clicks = None
    
    
    self.train = ClickDataSplit(data, clicks, 'train')
    self.valid = ClickDataSplit(data, clicks, 'valid')
    self.test = ClickDataSplit(data, clicks, 'test')
    self.click_data_split_by_name = {'train': self.train, 'valid': self.valid, 'test': self.test}
    
    self.max_ranklist_size = np.max(np.array([self.train.max_ranklist_size, self.valid.max_ranklist_size, self.test.max_ranklist_size]))
    self.train.max_ranklist_size = self.max_ranklist_size
    self.valid.max_ranklist_size = self.max_ranklist_size
    self.test.max_ranklist_size = self.max_ranklist_size
      
    
def read_click_data(dataset_name, datasets_info_path, data_fold, clicks_path):
  return ClickData(dataset_name, datasets_info_path, data_fold, clicks_path)