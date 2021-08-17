'''
Created on 29 Apr 2020

@author: aliv
'''

import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

class SoftmaxLoss(object):
  def __init__(self, topk):
    self._topk = topk
    
  def set_mask(self, mask):
    pass
  
  def loss_fn(self):
    def fn(y_true, y_pred, weights = None):
      y_pred = tf.reshape(y_pred, [-1, self._topk])
      if weights is None:
        y_w = y_true
      else:
        y_w = y_true * weights
        
      y_w += 1.e-12
      labels = (y_w) / tf.reduce_sum(y_w, axis=1, keepdims=True)
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_pred) * tf.reduce_sum(y_w,1)
#       loss = cross_entropy(logits=y_pred, labels=y_w)
      return tf.reduce_sum(loss) / tf.reduce_sum(y_true)
    return fn


class SigmoidLoss(object):
  def __init__(self, topk):
    self._topk = topk
    
  def set_mask(self, mask):
    pass
  
  def loss_fn(self):
    def fn(y_true, y_pred, weights = None):
      y_pred = tf.reshape(y_pred, [-1, self._topk])
      if weights is None:
        y_w = y_true
      else:
        y_w = y_true * weights
        
      labels = y_w
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_pred)
#       loss = cross_entropy(logits=y_pred, labels=y_w)
      return tf.reduce_sum(loss) / tf.reduce_sum(y_true)
    return fn

class LambdaLoss(object):
  def __init__(self, topk):
    self._topk = topk
    
  def set_mask(self, mask):
    self._ndoc = np.sum(mask, axis=-1).astype(np.int64)
#     print(mask)
  
  def loss_fn(self):
    def fn(y_true, y_pred, weights = None):
      y_pred = tf.reshape(y_pred, [-1, self._topk])
      if weights is None:
        y_w = y_true
      else:
        y_w = y_true * weights
        
      total_loss = np.array(0, dtype=np.float64)
      for i in range(y_true.shape[0]):
        total_loss += self.per_query_loss(y_w[i,:self._ndoc[i]], y_pred[i,:self._ndoc[i]])
      return total_loss / tf.reduce_sum(y_true)
    return fn
      
  def per_query_loss(self, y_w, y_pred):
    scores = y_pred.numpy()
    argsorted = np.argsort(-scores)
    ranked_positions = np.zeros_like(scores)
    ranked_positions[argsorted] = list(range(y_pred.shape[0]))
    
    gains = y_w
#       print(scores)
#       print(ranked_positions)
#       print(gains)
    
    gain_diff = gains[:, None] - gains[None, :]
    gain_mask = np.less_equal(gain_diff, 0.)
    
    rank_diff = np.abs(ranked_positions[:, None] - ranked_positions[None, :])
    rank_diff[gain_mask] = 1.
    
    disc_upp = 1. / np.log2(rank_diff+1.)
    disc_low = 1. / np.log2(rank_diff+2.)
    
    pair_w = disc_upp - disc_low
    pair_w *= np.abs(gain_diff)
    pair_w[gain_mask] = 0.
     
    score_diff = y_pred[:, None] - y_pred[None, :]
    loss = -tf.math.log((1./(1.+tf.math.exp(-score_diff)))**pair_w)/tf.cast(tf.math.log(2.), tf.float64)
    return tf.reduce_sum(loss)
    
        
        
        