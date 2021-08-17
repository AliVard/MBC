'''
Created on 16 Oct 2020

@author: aliv
'''
from .pbm_click_simulation import clean_and_sort
from ..correction.pbm_affine_correction import pad_and_reshape

import numpy as np

def get_oracle_weights(ranks, data_fold_split, topk, level2prob):
  argsorted, doclist_ranges = clean_and_sort(ranks, data_fold_split, topk)
  labels = data_fold_split.label_vector[argsorted]
  rel_probs = np.array(list(map(level2prob, labels)))
  
  rel_matrix = pad_and_reshape(doclist_ranges, topk, rel_probs)
  m = np.mean(rel_matrix, 0)
  weights = []
  for i in range(topk):
    weights.append([1.-m[i], m[i]])
    
  return weights