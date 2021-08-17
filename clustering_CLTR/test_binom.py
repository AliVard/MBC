'''
Created on 3 Jun 2020

@author: aliv
'''
import numpy as np
import pickle
from scipy.special import binom
from sklearn.utils.fixes import logsumexp


import matplotlib.pyplot as plt

with open('/Users/aliv/Dropbox/BitBucket/TrustBias/test_binom.pkl', 'rb') as f:
  saved_pickle = pickle.load(f)
  

def _estimate_weights(X, resp):
#   print(resp[:20])
  nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
  return nk / X.shape[0]

def _log_binomial_coeff(X):
  n_samples = X.shape[0]
  coeff = np.empty(n_samples)
  for i in range(n_samples):
    coeff[i] = np.log(binom(X[i,1],X[i,0]))
  return coeff

def _estimate_log_binomial_prob(X, ps):
  n_samples, _ = X.shape
  n_components = ps.shape[0]
  
  log_prob = np.empty((n_samples, n_components))
  
  p = np.log(np.finfo(np.float64).eps + ps) #todo: log(0.)
  q = np.log(1. - ps + np.finfo(np.float64).eps)
  
#   print(p)
#   print(q)
  coeff = _log_binomial_coeff(X)
#   print(coeff[:20].T)
#   print('---------')
  for c in range(n_components):
    log_prob[:, c] =  coeff + X[:,0] * p[c] + (X[:,1] - X[:,0]) * q[c]
    
#   print(log_prob[:20].T)
  return log_prob

def _estimate_weighted_log_prob(X, ps, weights):
  return _estimate_log_binomial_prob(X, ps) + np.log(weights)

        
def _estimate_log_prob_resp(X, ps):
  weights = np.ones_like(ps)
  weights /= weights.sum()
  
  for _ in range(10):
    weighted_log_prob = _estimate_weighted_log_prob(X, ps, weights)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under='ignore'):
      # ignore underflow
      log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
      
    weights = _estimate_weights(X, np.exp(log_resp))
    
  return weights

def compute_log_likelihood(click_count, click_shown, mask, pos, p1, p2):
  X = np.concatenate([click_count[mask[:,pos]==1,pos:pos+1], click_shown[mask[:,pos]==1,pos:pos+1]], 1)
  ps = np.array([p1, p2])
  weights = _estimate_log_prob_resp(X, ps)
  return np.sum(_estimate_weighted_log_prob(X, ps, weights)), weights

print(compute_log_likelihood(saved_pickle['count'], saved_pickle['shown'], saved_pickle['mask'], 0, 0.65, 0.98))
print(compute_log_likelihood(saved_pickle['count'], saved_pickle['shown'], saved_pickle['mask'], 0, 0.35, 0.98))
print(compute_log_likelihood(saved_pickle['count'], saved_pickle['shown'], saved_pickle['mask'], 0, 0.65, 0.78))
print(compute_log_likelihood(saved_pickle['count'], saved_pickle['shown'], saved_pickle['mask'], 0, 0.9, 0.65))