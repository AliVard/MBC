'''
Created on 5 Jun 2020

@author: aliv
'''
from .base import BaseCorrection, positions2ctr
from ..click_simulation.pbm_click_simulation import PBMClickSimulation

import numpy as np

from ..mLogger import mLoggers

def fullset_positions2ctr_1d(clicks, doclist_ranges):
  ctr_list = []
  for id in range(doclist_ranges.shape[0] - 1):
    ctr, n = positions2ctr(clicks[id], doclist_ranges[id+1] - doclist_ranges[id])
    if n == 0:
      n = 1
    ctr /= n
    ctr_list.append(ctr)
    
  return ctr_list

def pad_and_reshape(doclist_ranges, topk, vec):
  padded_vec = []
  for id in range(doclist_ranges.shape[0] - 1):
    e = doclist_ranges[id+1]
    if e > doclist_ranges[id] + topk:
      e = doclist_ranges[id] + topk
    padded_vec.append(vec[doclist_ranges[id]:e])
    if topk - (e - doclist_ranges[id]) > 0:
      padded_vec.append(np.zeros([topk - (e - doclist_ranges[id])], dtype = vec.dtype))
    
  return np.concatenate(padded_vec, 0).reshape([-1, topk])
  
def unpad(doclist_ranges, mat):
  vec = []
  for id in range(doclist_ranges.shape[0] - 1):
    vec.append(mat[id,:doclist_ranges[id+1]-doclist_ranges[id]])
  return np.concatenate(vec, 0)
  
class PBMAffineCorrection(BaseCorrection):
  def __init__(self, **kwargs):
    super(PBMAffineCorrection, self).__init__(**kwargs)
    
    self.train_ctr_list = None
    self.valid_ctr_list = None
    
    
  def _init_parameters(self):
    self.alpha = np.ones([self.topk], dtype=np.float64) - 1.e-3
    self.beta = np.zeros([self.topk], dtype=np.float64) + 1.e-6
    
    if self.params_kwargs is not None:
      if 'json' in self.params_kwargs:
        clickSimulation = PBMClickSimulation('', self.params_kwargs['json'], self.topk)
        clickSimulation._init_parameters()
        self.alpha = clickSimulation.alpha
        self.beta = clickSimulation.beta
      elif 'alpha' in self.params_kwargs:
        self.alpha = self.params_kwargs['alpha']
        self.beta = self.params_kwargs['beta']

    
    self.is_parameters_init = True
    

  def _correct_clicks(self, clicks, doclist_ranges):
    ctr_list = fullset_positions2ctr_1d(clicks, doclist_ranges)
      
    corrected_clicks = []
    for id in range(len(ctr_list)):
      ctr = ctr_list[id]
      corrected = (ctr - self.beta[:ctr.shape[0]])/self.alpha[:ctr.shape[0]]
      corrected[corrected < 0.] = 0.
#       corrected[corrected < 0.1] = 0.
      corrected[corrected > 20.] = 20.
#       corrected[(corrected > 0.9) & (corrected < 1.)] = 1.
#       
#       tmp = np.zeros_like(corrected)
#       tmp[corrected > 0.5] = 1.
      corrected_clicks.append(corrected)
      
    return np.concatenate(corrected_clicks, 0)
  
  def _E(self, rel_probs):
    r = rel_probs
    c = self.alpha * r + self.beta
    rc_dim = (2,2)
    rc_dim += r.shape
    rc = np.zeros(rc_dim, dtype=np.float64)
    
#     rc[1,1] > rc[1,0] -> zp / c > 1 > (1 - zp) / (1 - c)
    denom = np.array(1. - c)
    denom[denom == 0.] = 1.
    rc[1,0,:] = (r * (1. - (self.alpha + self.beta))) / denom
    rc[0,0,:] = 1. - rc[1,0,:]
    denom = np.array(c)
    denom[denom == 0.] = 1.
    rc[1,1,:] = r * (self.alpha + self.beta) / denom
    rc[0,1,:] = 1. - rc[1,1,:]
    
    return rc
  
  def init_EM(self):
    if self.train_ctr_list is None:
      self.train_ctr_list = fullset_positions2ctr_1d(self.clicks['train'], self.doclist_ranges['train'])
      self.train_mask = pad_and_reshape(self.doclist_ranges['train'], self.topk, np.ones_like(self.corrected_clicks['train']))
      self.c = pad_and_reshape(self.doclist_ranges['train'], self.topk, np.concatenate(self.train_ctr_list, 0))
    
    rel_probs = np.ones_like(self.c) * 0.5
    
    mLoggers['correction'].debug('{} -> {}'.format('clicks', 
                                   str(np.sum(self.c[:,:5]*self.train_mask[:,:5],0)/np.sum(self.train_mask[:,:5],0)).replace('\n',' ').replace(' ',',').replace(',,',',')))

    for _ in range(20):
      mLoggers['correction'].debug('{} -> {}'.format('rel_probs', 
                                   str(np.sum(rel_probs[:,:5]*self.train_mask[:,:5],0)/np.sum(self.train_mask[:,:5],0)).replace('\n',' ').replace(' ',',').replace(',,',',')))
    
      rc = self._E(rel_probs)
  
  #     rc[1,1] > c * rc[1,1] + (1-c) * rc[1,0] -> rc[1,1] > rc[1,0]
      nom = np.sum(self.c * rc[1,1,:] * self.train_mask, axis=0)
      denom = nom + np.sum(((1. - self.c) * rc[1,0,:] * self.train_mask), axis=0)
      denom[denom==0.] = 1.
      zetap = nom / denom
      
  #     rc[0,1] < c * rc[0,1] + (1-c) * rc[0,0] -> rc[0,1] < rc[0,0]
      nom = np.sum(self.c * rc[0,1,:] * self.train_mask, axis=0)
      denom = nom + np.sum(((1. - self.c) * rc[0,0,:] * self.train_mask), axis=0)
      denom[denom==0.] = 1.
      zetan = nom / denom
      
      self.alpha = zetap - zetan
      self.beta = zetan
      
      
      mLoggers['correction'].debug('zetap:{}'.format(list(self.alpha[:10] + self.beta[:10])))
      mLoggers['correction'].debug('zetan:{}'.format(list(self.beta[:10])))
      self.correct()
      rel_probs = pad_and_reshape(self.doclist_ranges['train'], self.topk, self.corrected_clicks['train'])
    
#       rel_probs = ((self.c * rc[1,1,:]) + ((1. - self.c) * rc[1,0,:])) * self.train_mask
      
    self.gamma = unpad(self.doclist_ranges['train'], ((self.c * rc[1,1,:]) + ((1. - self.c) * rc[1,0,:])) * self.train_mask)
    self.correct()
  
  def update(self, logits, logits_to_probs_fn):
    if self.train_ctr_list is None:
      self.train_ctr_list = fullset_positions2ctr_1d(self.clicks['train'], self.doclist_ranges['train'])
      self.train_mask = pad_and_reshape(self.doclist_ranges['train'], self.topk, np.ones_like(self.corrected_clicks['train']))
      self.c = pad_and_reshape(self.doclist_ranges['train'], self.topk, np.concatenate(self.train_ctr_list, 0))
    
    y_pred = pad_and_reshape(self.doclist_ranges['train'], self.topk, logits)
    
    rel_probs = logits_to_probs_fn(y_pred)
    
    mLoggers['correction'].debug('{} -> {}'.format('rel_probs', 
                                   str(np.sum(rel_probs[:,:5]*self.train_mask[:,:5],0)/np.sum(self.train_mask[:,:5],0)).replace('\n',' ').replace(' ',',').replace(',,',',')))
    mLoggers['correction'].debug('{} -> {}'.format('clicks', 
                                   str(np.sum(self.c[:,:5]*self.train_mask[:,:5],0)/np.sum(self.train_mask[:,:5],0)).replace('\n',' ').replace(' ',',').replace(',,',',')))

    if np.mean(np.sum(rel_probs[:,:5]*self.train_mask[:,:5],0)/np.sum(self.train_mask[:,:5],0)) > 0.8:
      self.alpha = None
      return
    for _ in range(2):
      rc = self._E(rel_probs)
  
  #     rc[1,1] > c * rc[1,1] + (1-c) * rc[1,0] -> rc[1,1] > rc[1,0]
      nom = np.sum(self.c * rc[1,1,:] * self.train_mask, axis=0)
      denom = nom + np.sum(((1. - self.c) * rc[1,0,:] * self.train_mask), axis=0)
      denom[denom==0.] = 1.
      zetap = nom / denom
      
  #     rc[0,1] < c * rc[0,1] + (1-c) * rc[0,0] -> rc[0,1] < rc[0,0]
      nom = np.sum(self.c * rc[0,1,:] * self.train_mask, axis=0)
      denom = nom + np.sum(((1. - self.c) * rc[0,0,:] * self.train_mask), axis=0)
      denom[denom==0.] = 1.
      zetan = nom / denom
      
      self.alpha = zetap - zetan
      self.beta = zetan
      
      
      mLoggers['correction'].debug('positive:{}'.format(list(self.alpha[:5] + self.beta[:5])))
      mLoggers['correction'].debug('negative:{}'.format(list(self.beta[:5])))
    
    self.gamma = unpad(self.doclist_ranges['train'], ((self.c * rc[1,1,:]) + ((1. - self.c) * rc[1,0,:])) * self.train_mask)
    self.gamma[self.gamma > 0.99] = 1.
    self.gamma[self.gamma < 0.01] = 0.
#     self.correct()
    
  def get_validation_gamma(self, logits, logits_to_probs_fn):
    if self.valid_ctr_list is None:
      self.valid_ctr_list = fullset_positions2ctr_1d(self.clicks['valid'], self.doclist_ranges['valid'])
      self.valid_mask = pad_and_reshape(self.doclist_ranges['valid'], self.topk, np.ones_like(self.corrected_clicks['valid']))
      self.valid_c = pad_and_reshape(self.doclist_ranges['valid'], self.topk, np.concatenate(self.train_ctr_list, 0))
    
    y_pred = pad_and_reshape(self.doclist_ranges['valid'], self.topk, logits)
    
    rel_probs = logits_to_probs_fn(y_pred)
    
    rc = self._E(rel_probs)
  
    self.valid_gamma = unpad(self.doclist_ranges['valid'], ((self.valid_c * rc[1,1,:]) + ((1. - self.valid_c) * rc[1,0,:])) * self.valid_mask)
    self.valid_gamma[self.valid_gamma > 0.99] = 1.
    self.valid_gamma[self.valid_gamma < 0.01] = 0.