'''
Created on 9 Jun 2020

@author: aliv
'''
import os
import numpy as np
import logging

logger = logging.getLogger('learning')

from .base import combine_corrected_clicks_and_features
from .lambdamart_learner import LambdamartLearner
from .DNN_learner import DNNLearner
from ..correction.pbm_affine_correction import PBMAffineCorrection
from ..correction.pbm_clustering_correction import PBMClusteringCorrection
from ..correction.full_info_correction import FullInfoCorrection
from ..correction.pbm_oracle_clustering_correction import PBMOracleClusteringCorrection
from .EM_learner import EMLearner

def min_max_prob(logits):
  e = np.exp(logits - np.max(logits,-1, keepdims=True))
  e = e - np.min(e,-1, keepdims=True)
  prob = 1. * e / np.max(e, -1, keepdims=True)
  prob[np.isnan(prob)] = 0.5
  return prob

def softmax_prob(logits):
  e = np.exp(logits - np.max(logits,-1, keepdims=True))
  prob = 1. * e / np.sum(e, -1, keepdims=True)
  prob[np.isnan(prob)] = 0.5
  return prob


def sigmoid_prob(logits):
#   x = (logits - np.mean(logits, -1, keepdims=True))
  x = logits
  return 1./(1. + np.exp(-x)) 

def sigmoid_zeromean_prob(logits):
  x = (logits - np.mean(logits, -1, keepdims=True))
#   x = logits
  return 1./(1. + np.exp(-x)) 

def train(data,
          train_clicks_pickle_path,
          correction_method,
          correction_kwargs,
          learning_algorithm,
          learning_kwargs):
  learner = {'lambdamart':LambdamartLearner,
             'DNN':DNNLearner
             }[learning_algorithm](**learning_kwargs)
             
  correction = {'pbm_affine':PBMAffineCorrection,
                'no_correction':PBMAffineCorrection,
                'pbm_soft_clustering':PBMClusteringCorrection,
#                 'pbm_soft2_clustering':PBMClusteringCorrection,
                'pbm_hard_clustering':PBMClusteringCorrection,
                'pbm_oracle_soft_clustering':PBMClusteringCorrection,
                'full_info':FullInfoCorrection
                }[correction_method](**correction_kwargs)
  
  combined = combine_corrected_clicks_and_features(train_clicks_pickle_path, data, correction)
  
  learner.train(combined.train, combined.valid)
  
  return learner


def train_EM(data,
          train_clicks_pickle_path,
          correction_method,
          regression_fn,
          regression_kwargs,
          EM_iterations,
          binary_rel):
  learner = {'lambdamart':LambdamartLearner,
             'DNN':DNNLearner
             }[regression_fn](**regression_kwargs)
             
  test_learner = {'lambdamart':None,
                  'DNN':LambdamartLearner(early_stopping_rounds=10000, eval_at=[10])
                  }[regression_fn]
  correction = {'pbm_affine':PBMAffineCorrection
                }[correction_method](**{'d':1})
  
  
  combined = combine_corrected_clicks_and_features(train_clicks_pickle_path, data, correction)
  
  EM_learner = EMLearner(learner, correction)
  logits_to_probs_fn = min_max_prob
  if regression_kwargs['loss_function_str'] == 'sigmoid':
    logits_to_probs_fn = sigmoid_prob
  elif regression_kwargs['loss_function_str'] == 'sigmoid_zeromean':
    logits_to_probs_fn = sigmoid_zeromean_prob
      
  result = EM_learner.EM_train(combined.train, combined.valid, 
                               iterations = EM_iterations, 
                               logits_to_probs_fn = logits_to_probs_fn, 
                               test_learner=test_learner, 
                               testset=data.test, 
                               binary_rel=binary_rel)
  
  return result

def test(data, learner, save_to, binary_rel):
  return learner.test(data.test, save_to=save_to, binary_rel=binary_rel)

def train_and_test_insidejob( data,
                              train_clicks_pickle_path,
                              correction_method,
                              correction_kwargs,
                              learning_algorithm,
                              learning_kwargs,
                              regression_fn,
                              regression_kwargs,
                              output_path,
                              slurm_job_id,
                              EM_iterations,
                              binary_rel):
  
  logger.info('''"slurm_job_id={}"
  "train_clicks_pickle_path={}"
  "correction_method={}"
  "correction_kwargs={}"
  "learning_algorithm={}"
  "learning_kwargs={}"
  "regression_fn={}"
  "regression_kwargs={}"
  '''.format(
    slurm_job_id,
    os.path.basename(train_clicks_pickle_path),
    correction_method,
    correction_kwargs,
    learning_algorithm,
    learning_kwargs,
    regression_fn,
    regression_kwargs))
  
  if learning_algorithm.startswith('EM_'):
    result = train_EM(data, train_clicks_pickle_path, correction_method, regression_fn, regression_kwargs, EM_iterations, binary_rel)
#     with open(output_path, 'a+') as f:
#       f.write('"slurm_job_id={}", "train_clicks_pickle_path={}", "correction_method={}", "correction_kwargs={}", "regression_fn={}", "regression_kwargs={}", "regression_best_ndcg={}", "regression_best_alpha={}", "regression_best_beta={}"\n'.
#               format(slurm_job_id,
#                      os.path.basename(train_clicks_pickle_path),
#                      correction_method,
#                      correction_kwargs,
#                      regression_fn,
#                      regression_kwargs,
#                      result['ndcg'],
#                      str(result['alpha']).replace('\n','').replace('  ',' ').replace(' ',','),
#                      str(result['beta']).replace('\n','').replace('  ',' ').replace(' ',',')))
#     learning_algorithm = learning_algorithm[3:]
    correction_kwargs = result
    np.set_printoptions(linewidth=np.inf)

  learner = train(data,
          train_clicks_pickle_path,
          correction_method,
          correction_kwargs,
          learning_algorithm.replace('EM_',''),
          learning_kwargs)
  ndcg, map_meas = test(data, 
                        learner, 
                        os.path.join(os.path.dirname(train_clicks_pickle_path),'outputs/{}.pkl'.format(slurm_job_id)),
                        binary_rel)
  print('click_file:{}, correction:{}, learning:{}, nDCG@10:{}'.format(
    os.path.basename(train_clicks_pickle_path), correction_method, learning_algorithm, ndcg))
  
  with open(output_path, 'a+') as f:
    f.write('"slurm_job_id={}", "train_clicks_pickle_path={}", "correction_method={}", "correction_kwargs={}", "learning_algorithm={}", "learning_kwargs={}", "regression_fn={}", "regression_kwargs={}", "EM_iterations={}", "binary_rel={}", "nDCG@10={}", "MAP={}"\n'.
            format(slurm_job_id,
                   os.path.basename(train_clicks_pickle_path),
                   correction_method,
                   str(correction_kwargs).replace('\n',' '),
                   learning_algorithm,
                   learning_kwargs,
                   regression_fn,
                   regression_kwargs,
                   EM_iterations,
                   binary_rel,
                   ndcg,
                   map_meas))
  
  
def train_and_test( data,
                    train_clicks_pickle_path,
                    correction_method,
                    correction_kwargs,
                    learning_algorithm,
                    learning_kwargs,
                    regression_fn,
                    regression_kwargs,
                    output_path,
                    slurm_job_id,
                    EM_iterations,
                    binary_rel):
  if learning_algorithm.startswith('EM_'):
    if isinstance(EM_iterations, str):
      EM_iterations = eval(EM_iterations)
    if isinstance(EM_iterations, int):
      EM_iterations = [EM_iterations]
    for EM_iteration in EM_iterations:
      train_and_test_insidejob(data, train_clicks_pickle_path, correction_method, correction_kwargs, learning_algorithm, 
                               learning_kwargs, regression_fn, regression_kwargs, output_path, slurm_job_id, 
                               EM_iteration, binary_rel)
  else:
    train_and_test_insidejob(data, train_clicks_pickle_path, correction_method, correction_kwargs, learning_algorithm, 
                             learning_kwargs, regression_fn, regression_kwargs, output_path, slurm_job_id, 
                             EM_iterations, binary_rel)
      
      
      
      
      
      
      
      
      
      
      
      
