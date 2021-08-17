'''
Created on 27 May 2020

@author: aliv
'''

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.mixture import GaussianMixture
import sys
import time
import os
import pickle
import lightgbm as lgb

from absl import app
from absl import flags

import matplotlib.pyplot as plt
from scipy import stats

from clustering_CLTR import metrics, common_functions
from clustering_CLTR.data_utils import read_click_data
from clustering_CLTR.binomial_mixture import BinomialMixture

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_string(  'dataset_name', 'Webscope_C14_Set1', 
                        'name of dataset: "MSLR-WEB30k" or "Webscope_C14_Set1"')
  flags.DEFINE_string(  'datasets_info_path', '/Users/aliv/eclipse-workspace/myModules/trust_bias/preprocess/datasets_info.json', 
                        'path to the datasets info file.')
  flags.DEFINE_string(  'clicks_path', '/Users/aliv/Dropbox/BitBucket/TrustBias/524288.trust_1_top50.lgbm_20_3.set1.pkl', 
                        'path to the clicks pickle file.')
  flags.DEFINE_integer( 'data_fold', 0, 
                        'data fold number')
  flags.DEFINE_integer( 'topk', 50, 
                        'data fold number')
  flags.DEFINE_string(  'model_dir', '/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/lgbm_20_3.txt', 
                        'with LambdaMART model directory for saving trained model.')
  flags.DEFINE_string(  'mixture', 'gaussian', 
                        'binomial or gaussian')
  
   
def get_zetas(ind):
  theta = '[(1./(i+1.)**{}) for i in range(20)]'.format(2 - ind % 2)
  ep = '[0.98-(i/100.) for i in range(20)]'
  en = '[0.{}5/(i+1.) for i in range(10)]'.format(6 if ind < 3 else 3)
  
  _theta = eval(theta)
  _ep = eval(ep)
  _en = eval(en)
  theta = np.zeros(50) + _theta[-1]
  ep = np.zeros(50) + _ep[-1]
  en = np.zeros(50) + _en[-1]
  theta[:len(_theta)] = _theta
  ep[:len(_ep)] = _ep
  en[:len(_en)] = _en
  
  return theta*ep, theta*en
  
def get_bin_from_levels(labels):
  bin_l = np.zeros_like(labels)
  bin_l[labels > 2] = 1.
  return bin_l

def lambdarank(train_dataset, valid_dataset, model_path, early_stopping_rounds):
  gbm = lgb.LGBMRanker()

  gbm.fit(train_dataset.feature_matrix, train_dataset.label_vector, 
          group=np.diff(train_dataset.doclist_ranges), 
          eval_set=[(valid_dataset.feature_matrix, valid_dataset.label_vector)],
          eval_group=[np.diff(valid_dataset.doclist_ranges)], 
          eval_at=[10], 
          early_stopping_rounds=early_stopping_rounds, 
          verbose=False)
  
  gbm.booster_.save_model(model_path)
  
  return gbm


def predict_by_model(model_path, X_test, y_test, q_test, eval_at):
  booster = lgb.Booster(model_file=model_path)
  y_pred = booster.predict(X_test)
  
  metric = metrics.LTRMetrics(y_test,q_test,y_pred)
  print('{} -> {}'.format(os.path.basename(model_path), [metric.NDCG(k) for k in eval_at]))
  
  ranks = -np.ones_like(y_pred, dtype=np.int64)
  pos = 0
  for cnt in q_test:
    session = y_pred[pos:pos+cnt]
    inds = session.argsort()[::-1]
    for i in range(len(inds)):
      ranks[pos+inds[i]] = i
    pos += cnt
  
  if len(ranks[ranks==-1]) > 0:
    raise Exception('ranks array has {} unassigned values!'.format(len(ranks[ranks==-1])))
  
  return ranks

def rate_to_labels(GMM, rate):
  y = GMM.predict(rate)
  argsorted = np.argsort(GMM.means_[:,0])
  true_y = np.zeros_like(y)
  for i in range(GMM.n_components):
    true_y[y==argsorted[i]]=i
    
#   print(list(rate[:40,0]))
#   print(list(true_y[:40]))
  return true_y
  
def fit_predict(n_components=2, n_init=1):
  click_data = read_click_data( dataset_name = FLAGS.dataset_name, 
                                datasets_info_path = FLAGS.datasets_info_path, 
                                data_fold = FLAGS.data_fold, 
                                clicks_path = FLAGS.clicks_path)
  
#   print('finished reading data!')
  

  _, rate, mask, click_count, click_shown = click_data.train.load_batch_rel_clicks(None)
  _, rate_v, mask_v, click_count_v, click_shown_v = click_data.valid.load_batch_rel_clicks(None)
  
  rate = rate.reshape([-1,FLAGS.topk])
  click_count = click_count.reshape([-1,FLAGS.topk])
  click_shown = click_shown.reshape([-1,FLAGS.topk])
  mask = mask.reshape([-1,FLAGS.topk])
  
  
  rate_v = rate_v.reshape([-1,FLAGS.topk])
  click_count_v = click_count_v.reshape([-1,FLAGS.topk])
  click_shown_v = click_shown_v.reshape([-1,FLAGS.topk])
  mask_v = mask_v.reshape([-1,FLAGS.topk])
  
  train_set = type('', (), {})()
  train_set.feature_matrix = click_data.train.feature_matrix
  train_set.doclist_ranges = click_data.train.doclist_ranges
  
  valid_set = type('', (), {})()
  valid_set.feature_matrix = click_data.valid.feature_matrix
  valid_set.doclist_ranges = click_data.valid.doclist_ranges
  
  labels = np.zeros_like(rate)
  labels_v = np.zeros_like(rate_v)
  
  zp, zn = get_zetas(1)
  for pos in range(FLAGS.topk):
    GMM = {'binomial':BinomialMixture, 
           'gaussian':GaussianMixture}[FLAGS.mixture](n_components=n_components, n_init=n_init, warm_start=True, max_iter=100)
    if FLAGS.mixture == 'gaussian':
      train_X = rate[mask[:,pos]==1,pos:pos+1]
      valid_X = rate_v[mask_v[:,pos]==1,pos:pos+1]
    elif FLAGS.mixture == 'binomial':
      train_X = np.concatenate([click_count[mask[:,pos]==1,pos:pos+1], click_shown[mask[:,pos]==1,pos:pos+1]], 1)
      valid_X = np.concatenate([click_count_v[mask_v[:,pos]==1,pos:pos+1], click_shown_v[mask_v[:,pos]==1,pos:pos+1]], 1)
      
#     print('----------------------------------------------------------------------------')
#     print('pos {} -> {} and {}, length: {}'.format(pos, zn[pos], zp[pos], train_X.shape[0]))
    for _ in range(1):
      GMM.fit(train_X)
#       w,m = GMM._get_printable_parameters()
#       print('w:{}, m:{}'.format(list(w), list(m)))
#       w,m,c,ch = GMM._get_parameters()
#       print('w:{}, m:{}, c:{}, ch:{}'.format(list(w), list(m), list(c), list(ch)))
#     print(GMM.means_)
#     print(GMM.covariances_)
#     print(zn[pos] * (1. - zn[pos]) / len(mask[:,pos]==1) * 2)
#     GMM.means_[GMM.means_[:,0].argmin()] = zn[pos]
#     GMM.means_[GMM.means_[:,0].argmax()] = zp[pos]
#     GMM.covariances_[GMM.means_[:,0].argmin()] = zn[pos] * (1. - zn[pos]) / len(mask[:,pos]==1) / 0.8
#     GMM.covariances_[GMM.means_[:,0].argmax()] = zp[pos] * (1. - zp[pos]) / len(mask[:,pos]==1) / 0.2
#     print(GMM.means_)
#     print(GMM.covariances_)
    labels[mask[:,pos]==1,pos] = rate_to_labels(GMM, train_X)
    labels_v[mask_v[:,pos]==1,pos] = rate_to_labels(GMM, valid_X)
    
  
  train_set.label_vector = labels[mask==1]
  error = get_bin_from_levels(click_data.train.label_vector) - train_set.label_vector
  print('equal: {}, +1:{}, -1:{}'.format(len(error[error==0]), len(error[error==1]), len(error[error==-1])))
  valid_set.label_vector = labels_v[mask_v==1]
  
#   print('training ...')
  gbm = lambdarank(train_set, valid_set, FLAGS.model_dir, 10000)
    
#   print('training ideal ...')
#   train_set.label_vector = np.zeros_like(click_data.train.label_vector)
#   valid_set.label_vector = np.zeros_like(click_data.valid.label_vector)
#   train_set.label_vector[click_data.train.label_vector>2] = 1
#   valid_set.label_vector[click_data.valid.label_vector>2] = 1
#   gbm_ideal = lambdarank(train_set, valid_set, FLAGS.model_dir, 10000)
    
#   print('predicting ...')
  y_pred = gbm.booster_.predict(click_data.test.feature_matrix)
#   y_pred_ideal = gbm_ideal.booster_.predict(click_data.test.feature_matrix)
  lv = np.zeros_like(click_data.test.label_vector)
  lv[click_data.test.label_vector>2] = 1
  metric = metrics.LTRMetrics(lv,np.diff(click_data.test.doclist_ranges),y_pred)
  return metric.NDCG(10)
#   metric = metrics.LTRMetrics(lv,np.diff(click_data.test.doclist_ranges),y_pred_ideal)
#   print('ideal: {}'.format(metric.NDCG(10)))
  
def main(args):
  for count in [17,19,21,23]:
    click_count = str(2**count)
    FLAGS.clicks_path = '/Users/aliv/Dropbox/BitBucket/TrustBias/{}.trust_2_top50.lgbm_20_3.set1.pkl'.format(click_count)
    ndcg = fit_predict()
    print('2**%d  %f' %(count, ndcg))

  click_data = read_click_data( dataset_name = FLAGS.dataset_name, 
                                datasets_info_path = FLAGS.datasets_info_path, 
                                data_fold = FLAGS.data_fold, 
                                clicks_path = FLAGS.clicks_path)
  
#   print('finished reading data!')
  

#   _, rate, mask, click_count, click_shown = click_data.train.load_batch_rel_clicks(None)
#   
#   rate = rate.reshape([-1,FLAGS.topk])
#   click_count = click_count.reshape([-1,FLAGS.topk])
#   click_shown = click_shown.reshape([-1,FLAGS.topk])
#   mask = mask.reshape([-1,FLAGS.topk])
#   save_pickle = {}
#   save_pickle['count'] = click_count
#   save_pickle['shown'] = click_shown
#   save_pickle['mask'] = mask
#   save_pickle['rate'] = rate
#   with open('/Users/aliv/Dropbox/BitBucket/TrustBias/test_binom.pkl', 'wb') as f:
#     pickle.dump(save_pickle, f, protocol=2)
  
  
if __name__ == '__main__':
  app.run(main)
  
  