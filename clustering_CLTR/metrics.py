'''
Created on Fri Mar  6  2020

@author: aliv
'''
import numpy as np
from math import log
import os

from absl import app
from absl import flags

from sklearn.datasets import load_svmlight_file

from .mLogger import mLoggers

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  
  
  flags.DEFINE_integer( 'topk', 10, '')
  flags.DEFINE_string(  'path', '/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/ltrc_yahoo_small/Data/set1.train.b.lgbm_800_2.top20.txt', '')
  flags.DEFINE_bool(    'svm', True, 'is the file in the svmlight format or just a two column prediction file.')
  flags.DEFINE_list(    'eval_at', ['1', '3', '5', '10'], 'list of "k" values for evaluating NDCG@k.')

class LTRMetrics:
  def __init__(self, y, query_count, y_pred = None, ranks = None, topk = 50):
#     self._y = y
#     self._y_pred = y_pred
    self._query_count = np.cumsum(np.array(query_count), axis=0)
    
    self._querySeparatedMap = {}
    pos = 0
    for i, cnt in enumerate(query_count):
      tmp_y = np.array(y[pos:pos+cnt], copy=True)
      if ranks is not None:
        y_snapshot = tmp_y[ranks[pos:pos+cnt] < topk]
        tmp_y = y_snapshot
      
      if y_pred is not None:
        y_snapshot = y_pred[pos:pos+cnt]
        if ranks is not None:
          tmp = y_snapshot[ranks[pos:pos+cnt] < topk]
          y_snapshot = tmp
        tmp_y = tmp_y[y_snapshot.argsort()[::-1]]
        
#       jointList = np.array(list(zip(y_pred[pos:pos+cnt],y[pos:pos+cnt])))
#       self._querySeparatedMap[i] = jointList[jointList[:,0].argsort()[::-1]]
      tmp_y = np.array(tmp_y)
      self._querySeparatedMap[i] = tmp_y
      pos += cnt
      
  def MAP(self):
    map = 0.
    denom = 0
    for _,docs in self._querySeparatedMap.items():
      rels = 0.
      ap = 0.
      for i in range(len(docs)):
        if docs[i] == 1:
          rels += 1.
          ap += rels / (i + 1.)
      if rels > 0:
        map += ap / rels
        denom += 1
    if denom > 0:
      return map / denom
    else:
      return -1
    
  def DCG(self, k):
    dcg = 0.
    for _,docs in self._querySeparatedMap.items():
      effective_k = k
      if k > len(docs):
        effective_k = len(docs)

      for i in range(effective_k):
#         dcg += (2**docs[i,1]-1)/log2(i+1+1)
        dcg += (2**docs[i]-1)/log(i+1+1, 2)
        
    return dcg/len(self._querySeparatedMap)
#   
#   def unbiased_DCG(self, k, weights):
#     dcg = 0
#     for _,docs in self._querySeparatedMap.items():
#       if k > len(docs):
#         k = len(docs)
#       for i in range(k):
# #         dcg += (2**docs[i,1]-1)/log2(i+1+1)
#         dcg += weights[i] * (2**docs[i]-1)/log(i+1+1, 2)
#         
# #       print(dcg)
#         
#     return dcg/len(self._querySeparatedMap)
  
#   def affine_DCG(self, k, weights, noise):
#     dcg = 0
#     for qid___,docs in self._querySeparatedMap.items():
#       if k > len(docs):
#         k = len(docs)
#       for i in range(k):
# #         dcg += (2**docs[i,1]-1)/log2(i+1+1)
#         dcg += weights[i] * (2**docs[i]-1-noise[i])/log(i+1+1, 2)
#         
# #       print(dcg)
#         
#     return dcg/len(self._querySeparatedMap)
  
  def NDCG(self, k):
#     zero_dcg = 0
    ndcg = 0
    denum = 0
    sum_dcg = []
    for qid___,docs in self._querySeparatedMap.items():
      effective_k = k
      if k > len(docs):
        effective_k = len(docs)
      
      dcg = 0
      idcg = 0
      for i in range(effective_k):
#         dcg += (2**docs[i,1]-1)/log2(i+1+1)
        dcg += (2**docs[i]-1)/log(i+1+1, 2)
        
#       if dcg == 0:
#         zero_dcg += 1
      docs_ = np.array(docs[:],copy=True)
#       docs_ = np.array(docs[:,1],copy=True)
      docs_.sort()
      docs_ = docs_[::-1]
      for i in range(effective_k):
        idcg += (2**docs_[i]-1)/log(i+1+1, 2)
        
      if idcg > 0:
        ndcg += dcg/idcg
        denum += 1
        sum_dcg.append(dcg/idcg)
#       if dcg > 4.5:
#         print(docs[:10])
#         if dcg/idcg > 0.9:
#           print('id:{}, line:{}~{}, ratio:{}'.format(qid___, self._query_count[qid___], self._query_count[qid___+1], dcg/idcg))
    
    return ndcg/denum
  
  
  
  def NDCG_perquery(self, k):
#     zero_dcg = 0
    ndcg = []
    for qid___,docs in self._querySeparatedMap.items():
      effective_k = k
      if k > len(docs):
        effective_k = len(docs)
      
      dcg = 0
      idcg = 0
      for i in range(effective_k):
#         dcg += (2**docs[i,1]-1)/log2(i+1+1)
        dcg += (2**docs[i]-1)/log(i+1+1, 2)
        
#       if dcg == 0:
#         zero_dcg += 1
      docs_ = np.array(docs[:],copy=True)
#       docs_ = np.array(docs[:,1],copy=True)
      docs_.sort()
      docs_ = docs_[::-1]
      for i in range(effective_k):
        idcg += (2**docs_[i]-1)/log(i+1+1, 2)
        
      if idcg > 0:
        ndcg.append(dcg/idcg)
      else:
        ndcg.append(-1)
    
    return ndcg
  
  
  def queryCount(self):
    return len(self._querySeparatedMap)
  
def eval_output(y_true, y_pred, query_counts, report_dcg, k, ranks=None, topk=50):
  if isinstance(query_counts, int):
    query_counts = np.ones([int(len(y_pred)/query_counts)],
                           dtype=np.int)*query_counts

  ltr = LTRMetrics(y_true,query_counts,y_pred, ranks, topk)
  
  if not report_dcg:
    return ltr.NDCG(k)
  else:
    return ltr.NDCG(k), ltr.DCG(k)
  
def eval_output_unbiased(y_true, y_pred, query_counts, weights, k):
  if isinstance(query_counts, int):
    query_counts = np.ones([int(len(y_pred)/query_counts)],
                           dtype=np.int)*query_counts

  ltr = LTRMetrics(y_true,query_counts,y_pred)
  
  return ltr.unbiased_DCG(int(k), weights)
  
def eval_output_unbiased_denoised(y_true, y_pred, query_counts, weights, noise, k):
  if isinstance(query_counts, int):
    query_counts = np.ones([int(len(y_pred)/query_counts)],
                           dtype=np.int)*query_counts

  ltr = LTRMetrics(y_true,query_counts,y_pred)
  
  return ltr.affine_DCG(int(k), weights, noise)
  
def eval_predictions(path, eval_at, query_counts=10):
  predicts = np.genfromtxt(path, delimiter=',')
  y_pred = predicts[:,0]
  y_true = predicts[:,1]
  if isinstance(query_counts, int):
    query_counts = np.ones([int(len(y_pred)/query_counts)],
                           dtype=np.int)*query_counts

  ltr = LTRMetrics(y_true,query_counts,y_pred)
  ltr_orig = LTRMetrics(y_true,query_counts)
  
  print('{} -> {}'.format(os.path.basename(path), [ltr.NDCG(int(k)) for k in eval_at]))
  print('original -> {}'.format([ltr_orig.NDCG(int(k)) for k in eval_at]))

def eval_svmlight(file_path, eval_at):
  _, y, q = load_svmlight_file(file_path, query_id=True)
  _, q = np.unique(q,return_counts=True)
  ltr = LTRMetrics(y,q)

  print('{} -> {}'.format(os.path.basename(file_path), [ltr.NDCG(int(k)) for k in eval_at]))
    
def main(argv):
  if FLAGS.svm:
    eval_svmlight(FLAGS.path, FLAGS.eval_at)
  else:
    eval_predictions(FLAGS.path, FLAGS.eval_at, FLAGS.topk)
  
if __name__ == '__main__':
  app.run(main)
