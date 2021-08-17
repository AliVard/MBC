'''
Created on 18 Nov 2019

@author: aliv
'''

import lightgbm as lgb
import numpy as np
import os
import sys
import pickle


from absl import app
from absl import flags


SYS_PATH_APPEND_DEPTH = 3
SYS_PATH_APPEND = os.path.abspath(__file__)
for _ in range(SYS_PATH_APPEND_DEPTH):
  SYS_PATH_APPEND = os.path.dirname(SYS_PATH_APPEND)
sys.path.append(SYS_PATH_APPEND)

from clustering_CLTR import metrics, common_functions
from clustering_CLTR import dataset

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  
  flags.DEFINE_string(  'dataset_name', 'Webscope_C14_Set1', 
#   flags.DEFINE_string(  'dataset_name', 'MSLR-WEB30k', 
                        'name of dataset: "MSLR-WEB30k" or "Webscope_C14_Set1"')
  flags.DEFINE_string(  'datasets_info_path', '/Users/aliv/eclipse-workspace/myModules/trust_bias/preprocess/datasets_info.json', 
                        'path to the datasets info file.')
  flags.DEFINE_integer( 'data_fold', 0, 'data fold number')
  
  flags.DEFINE_list(    'train_query_count_list', [1,1,1], 'number of queries used for training and validation')
  
  
#   flags.DEFINE_string(  'model_dir', '/Users/aliv/MySpace/_DataSets/LTR/Microsoft/MSLR-WEB30K/lgbm_0_0.txt', 
  flags.DEFINE_string(  'model_dir', '/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/lgbm_1_*', 
                        'with --nopredict (default): LambdaMART model directory for saving trained model.'
                        'with --predict: LambdaMART model directory for loading all trained models named "lgbm_*.txt" inside.')
  
  flags.DEFINE_integer( 'early_stopping_rounds', 1000,
                        'early_stopping_rounds for lightgbm\'s LGBMRanker().fit() function.')
  
  flags.DEFINE_bool(    'predict', True,
                        'train ("predict"=False) or test("predict"=True)')
  flags.DEFINE_list(    'eval_at', ['5','10','30'],
                        'list of "k" values for evaluating NDCG@k.')
  flags.DEFINE_integer( 'pickle_dump_protocol', 2,
                        'protocol number for pickle dump.')

_G_BINARIZE = False

def lambdarank(train_dataset, valid_dataset, model_path, early_stopping_rounds):
  gbm = lgb.LGBMRanker()
#   gbm.fit(X_train, y_train, group=q_train, eval_set=[(X_test, y_test)],
#           eval_group=[q_test], eval_at=[1, 3], early_stopping_rounds=100, verbose=False,
#           callbacks=[lgb.reset_parameter(learning_rate=lambda x: max(0.01, 0.1 - 0.01 * x))])
  
  if _G_BINARIZE:
    tl = np.zeros_like(train_dataset.label_vector)
    tl[train_dataset.label_vector>2] = 1
    vl = np.zeros_like(valid_dataset.label_vector)
    vl[valid_dataset.label_vector>2] = 1
  else:
    tl = train_dataset.label_vector
    vl = valid_dataset.label_vector
  gbm.fit(train_dataset.feature_matrix, tl, 
          group=np.diff(train_dataset.doclist_ranges), 
          eval_set=[(valid_dataset.feature_matrix, vl)],
          eval_group=[np.diff(valid_dataset.doclist_ranges)], 
          eval_at=[10], 
          early_stopping_rounds=early_stopping_rounds, 
          verbose=False)
  
  gbm.booster_.save_model(model_path)
  
  return gbm
  
def train(data, modelDir, samples_size, early_stopping_rounds):
  if not os.path.exists(modelDir):
    os.makedirs(modelDir)
    
  file_number = 0
  
  while os.path.exists(os.path.join(modelDir,'lgbm_{}_{}.txt'.format(samples_size, file_number))):
    file_number += 1

  model_path = os.path.join(modelDir,'lgbm_{}_{}.txt'.format(samples_size, file_number))
  if samples_size > 0:
    subsampled_train_data = data.train.random_subsample(samples_size)
    subsampled_valid_data = data.valid.random_subsample(samples_size)
  else:
    subsampled_train_data = data.train
    subsampled_valid_data = data.valid
    
  
  lambdarank(subsampled_train_data, subsampled_valid_data, model_path, early_stopping_rounds)
  print('finished training {} model!'.format(model_path))


  
def predict_all(data, model_dir, eval_at):
  test_path = data.train.data_raw_path
  test_dir = os.path.dirname(test_path)
  test_basename = os.path.basename(test_path)
  assert test_basename.endswith('train.txt'), 'what happend!'
  test_basename = test_basename[:-len('train.txt')]
  if len(test_basename) == 0:
    test_basename = "_"
  elif test_basename[-1] == '.':
    test_basename = test_basename[:-1]
  
  data_dir, data_files = common_functions.get_files(model_dir)
  
  for file in data_files:
    if os.path.isfile(os.path.join(data_dir, file)):
      if str.startswith(file,'lgbm_'):
        try:
          output_path = os.path.join(os.path.dirname(test_path), 'ranks.' + os.path.splitext(file)[0] + '.' + test_basename + '.pkl')
          all_ranks = {}
          for dfs in [data.train, data.valid, data.test]:
            if _G_BINARIZE:
              lv = np.zeros_like(dfs.label_vector, dtype=np.float64)
              lv[dfs.label_vector>2] = 1.
            else:
              lv = dfs.label_vector
            all_ranks[dfs.name] = predict_by_model(model_path = os.path.join(data_dir, file), 
                                                   X_test = dfs.feature_matrix, 
                                                   y_test = lv, 
                                                   q_test = np.diff(dfs.doclist_ranges), 
                                                   eval_at = eval_at)
          with open(output_path, 'wb') as f:
            pickle.dump(all_ranks, f, protocol=FLAGS.pickle_dump_protocol)
            
        except:
          print('except')
          pass

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
    
def main(args):
  data = dataset.get_dataset_from_json_info(
                    FLAGS.dataset_name,
                    FLAGS.datasets_info_path,
                  ).get_data_folds()[FLAGS.data_fold]
  data.read_data()
  
  if FLAGS.predict:
    predict_all(data, FLAGS.model_dir, list(map(int,FLAGS.eval_at)))
  else:
    for samples_size in list(map(int,FLAGS.train_query_count_list)):
      train(data, FLAGS.model_dir, samples_size, FLAGS.early_stopping_rounds)
    
if __name__ == '__main__':
  app.run(main)
  