'''
Created on 5 Jun 2020

@author: aliv
'''
from .base import BaseCorrection, positions2ctr
from .pbm_affine_correction import pad_and_reshape

import numpy as np

import logging

logger = logging.getLogger('correction')

def fullset_positions2ctr_1d(clicks, doclist_ranges):
  ctr_list = []
  for id in range(doclist_ranges.shape[0] - 1):
    ctr, n = positions2ctr(clicks[id], doclist_ranges[id+1] - doclist_ranges[id])
    if n == 0:
      n = 1
    ctr /= n
    ctr_list.append(ctr)
    
  return ctr_list

  
'''
true_labels = pad_and_reshape(affine_correction.doclist_ranges['train'], 50, data.train.label_vector[affine_correction.argsorted['train']])
true_bin_labels = np.zeros_like(true_labels)
true_bin_labels[true_labels > 2] = 1.

true_labels_v = pad_and_reshape(affine_correction.doclist_ranges['valid'], 50, data.valid.label_vector[affine_correction.argsorted['valid']])
true_bin_labels_v = np.zeros_like(true_labels_v)
true_bin_labels_v[true_labels_v > 2] = 1.

true_ctr_list = get_ctr_list(affine_correction)
'''
class FullInfoCorrection(BaseCorrection):
  def __init__(self, **kwargs):
    super(FullInfoCorrection, self).__init__(**kwargs)
    
    self.train_ctr_list = None
    
#     and "self.doclist_ranges" contains "self.argsorted"
  '''
  *** IMPORTANT ***
  in FullInfoCorrection, we abuse the "self.clicks" and "self.doclist_ranges"
  Here, we replace "clicks" with data.train.label_vector (or test, valid) 
  '''
  def _init_parameters(self):
    data = self.params_kwargs['data']
    self.lever2prob = eval(self.params_kwargs['json']['level2prob'])
    
    self.clicks['train'] = data.train.label_vector
    if 'valid' in self.clicks:
      self.clicks['valid'] = data.valid.label_vector
    if 'test' in self.clicks:
      self.clicks['test'] = data.test.label_vector
#       
#     self.doclist_ranges_bu = self.doclist_ranges
#     self.doclist_ranges = self.argsorted
    self.is_parameters_init = True
    
# and "doclist_ranges" contains "self.argsorted['train']" (or test, valid)
  '''
  *** IMPORTANT ***
  in FullInfoCorrection, we abuse the "clicks" and "doclist_ranges" arguments.
  Here, "clicks" contains data.train.label_vector (or test, valid) 
  '''
  def _correct_clicks(self, clicks, doclist_ranges):
    for name in ['train','test','valid']:
      if name in self.doclist_ranges and self.doclist_ranges[name] is doclist_ranges:
        break
    true_labels = clicks[self.argsorted[name]]
#     print('true: {} -> {}'.format(true_labels.shape, true_labels[:30]))
#     tmp=self.lever2prob(true_labels)
#     print('prob: {} -> {}'.format(tmp.shape, tmp[:30]))
    
    return self.lever2prob(true_labels)
  