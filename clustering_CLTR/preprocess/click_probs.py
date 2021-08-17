'''
Created on 2 Apr 2020

@author: aliv
'''

import numpy as np
import os
import sys
import pickle
import json
import time


from absl import app
from absl import flags


SYS_PATH_APPEND_DEPTH = 3
SYS_PATH_APPEND = os.path.abspath(__file__)
for _ in range(SYS_PATH_APPEND_DEPTH):
  SYS_PATH_APPEND = os.path.dirname(SYS_PATH_APPEND)
sys.path.append(SYS_PATH_APPEND)

from clustering_CLTR.preprocess import dataset

# python /Users/aliv/eclipse-workspace/myModules/trust_bias/preprocess/click_probs.py --datasets_info_path=/Users/aliv/eclipse-workspace/myModules/trust_bias/preprocess/datasets_info.json --click_policy_path=/Users/aliv/eclipse-workspace/myModules/trust_bias/preprocess/click_policy.json --ranks_pickle_path=/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/ranks.lgbm_20_3.set1.pkl

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  
#   flags.DEFINE_string(  'dataset_name', 'MSLR-WEB30k', 
  flags.DEFINE_string(  'dataset_name', 'Webscope_C14_Set1', 
                        'name of dataset: "MSLR-WEB30k" or "Webscope_C14_Set1"')
  flags.DEFINE_string(  'datasets_info_path', 'datasets_info.json', 
                        'path to the datasets info file.')
  flags.DEFINE_integer( 'data_fold', 0, 
                        'data fold number')
  
  flags.DEFINE_string(  'click_policy_path', 'click_policy.json', 
                        'path to the file containing the policy for simulating clicks.')
#   flags.DEFINE_string(  'ranks_pickle_path', '/Users/aliv/MySpace/_DataSets/LTR/Microsoft/MSLR-WEB30K/clicks/ranks.lgbm_20_3._.pkl',
  flags.DEFINE_string(  'ranks_pickle_path', '/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/ranks.lgbm_20_3.set1.pkl',
                        'path to the pickle file containing ranks of docs.')
  
  flags.DEFINE_integer( 'default_topk', 1000,
                        'the default value for topk when it is not specified in the click policy file.')
  
  flags.DEFINE_integer( 'pickle_dump_protocol', 2,
                        'protocol number for pickle dump.')
  
def load_click_policies(click_policy_path):
  with open(click_policy_path) as f:
    click_policies_dict = json.load(f)
  click_policies = {}
  for name in click_policies_dict:
    if name == 'comment':
      continue
    if click_policies_dict[name]['type'] == 'pbm':
      click_policies[name] = PBMTypeClicks(click_policies_dict[name])
    else:
      raise Exception('only pbm type is implemented currently!')
  return click_policies
    
class ClickProbsSplit:
  def __init__(self, all_probs, split_name):
    self.name = split_name
    self.probs = all_probs[ClickProbsSplit.probs_key(split_name)]
    self.ranks = all_probs[ClickProbsSplit.ranks_key(split_name)]
    
  @staticmethod
  def probs_key(name):
    return name + '_probs'
  @staticmethod
  def ranks_key(name):
    return name + '_ranks'

class ClickProbs:
  def __init__(self, pickle_path):
    with open(pickle_path, 'rb') as f:
      all_probs = pickle.load(f)
    
    self.train = ClickProbsSplit(all_probs, 'train')
    self.valid = ClickProbsSplit(all_probs, 'valid')
    self.test = ClickProbsSplit(all_probs, 'test')
    self.click_probss_split_by_name = {'train': self.train, 'valid': self.valid, 'test': self.test}
    
def load_click_probs_pickle(pickle_path):
  return ClickProbs(pickle_path)

class PBMTypeClicks:
  def __init__(self, spec_dict):
    assert 'type' in spec_dict, 'click policies should have "type"'
    assert spec_dict['type']=='pbm', 'PBMTypeClicks can only be built from a "type":"pbm" policy'
    
    theta = eval(spec_dict['theta'])
    epsilon_p = eval(spec_dict['epsilon+'])
    epsilon_n = eval(spec_dict['epsilon-'])
    level2prob = lambda x: float(x)
    if 'level2prob' in spec_dict:
      level2prob = eval(spec_dict['level2prob'])
      
    topk = FLAGS.default_topk
    if 'topk' in spec_dict:
      topk = spec_dict['topk']
    
    self.theta = np.zeros(topk) + theta[-1]
    self.epsilon_p = np.zeros(topk) + epsilon_p[-1]
    self.epsilon_n = np.zeros(topk) + epsilon_n[-1]
    self.theta[:len(theta)] = theta
    self.epsilon_p[:len(epsilon_p)] = epsilon_p
    self.epsilon_n[:len(epsilon_n)] = epsilon_n
    self.topk = topk
    self.level2prob = level2prob
    
    their_ips = (1. / self.theta) * (self.epsilon_p / (self.epsilon_n + self.epsilon_p))
    our_ips = 1. / (self.theta * (self.epsilon_p - self.epsilon_n))
    our_noise = self.theta * self.epsilon_n
    max_len = np.maximum(len(epsilon_n), len(epsilon_n))
    max_len = np.maximum(max_len, len(theta))
    print('\t\t"their_ips": "{}"'.format(str(list(their_ips[:max_len])).replace(' ','')))
    print('\t\t"our_ips": "{}"'.format(str(list(our_ips[:max_len])).replace(' ','')))
    print('\t\t"our_noise": "{}"'.format(str(list(our_noise[:max_len])).replace(' ','')))
    
  def _doc_click_prob(self, label_prob, rank):
    if rank >= self.topk:
      return -1.
    theta = self.theta[rank]
    epsilon_p = self.epsilon_p[rank]
    epsilon_n = self.epsilon_n[rank]

    if label_prob == 1.:
      return theta * epsilon_p
    elif label_prob == 0.:
      return theta * epsilon_n
    else:
      raise Exception('only binary relevance implemented! {} not recognized'.format(label_prob))
    
  def _query_click_probs(self, labels, ranks):
    label_probs = np.array(list(map(self.level2prob, labels)))
    # if no relevant doc within topk
#     if np.sum(label_probs[ranks<self.topk]) == 0.:
#       return -np.ones_like(label_probs, dtype=np.float64)
    
    probs = np.zeros_like(label_probs, dtype=np.float64)
    for i in range(len(label_probs)):
      probs[i] = self._doc_click_prob(label_probs[i], ranks[i])
    return probs
  
  def get_click_probs(self, labels, ranks, doclist_ranges):
    probs = []
    
    for qid in range(doclist_ranges.shape[0] - 1):
      s_i = doclist_ranges[qid]
      e_i = doclist_ranges[qid + 1]
      probs.append(self._query_click_probs(labels[s_i:e_i], ranks[s_i:e_i]))
      
    return np.concatenate(probs, axis=0)

def main(args):
  start_time = time.time()
  data = dataset.get_dataset_from_json_info(
                    FLAGS.dataset_name,
                    FLAGS.datasets_info_path,
                  ).get_data_folds()[FLAGS.data_fold]
  data.read_data()
  print('read data in {} seconds.'.format(time.time() - start_time))
  
  click_policies = load_click_policies(FLAGS.click_policy_path)
  with open(FLAGS.ranks_pickle_path, 'rb') as f:
    all_ranks = pickle.load(f)
  
  
  ranks_file_name = os.path.basename(FLAGS.ranks_pickle_path)
  assert str.startswith(ranks_file_name, 'ranks.'), 'where is "rank." prefix?'
  
  pickles_dir = os.path.dirname(FLAGS.ranks_pickle_path)
  
  for name in click_policies:
    start_time = time.time()
    clicks_pickle_path = os.path.join(pickles_dir, name + ranks_file_name[len('ranks'):])
    all_probs = {}
    for dfs in [data.train, data.valid, data.test]:
      all_probs[ClickProbsSplit.probs_key(dfs.name)] = click_policies[name].get_click_probs(dfs.label_vector, all_ranks[dfs.name], dfs.doclist_ranges)
      all_probs[ClickProbsSplit.ranks_key(dfs.name)] = all_ranks[dfs.name]
      
    with open(clicks_pickle_path, 'wb') as f:
      pickle.dump(all_probs, f, protocol=FLAGS.pickle_dump_protocol)
    print('dumped {}. took {} seconds.'.format(name, time.time() - start_time))

if __name__ == '__main__':
  app.run(main)