'''
Created on 2 Apr 2020

@author: aliv
'''

import numpy as np
import os
import sys
import pickle
import time


from absl import app
from absl import flags


SYS_PATH_APPEND_DEPTH = 3
SYS_PATH_APPEND = os.path.abspath(__file__)
for _ in range(SYS_PATH_APPEND_DEPTH):
  SYS_PATH_APPEND = os.path.dirname(SYS_PATH_APPEND)
sys.path.append(SYS_PATH_APPEND)

from clustering_CLTR import dataset
from clustering_CLTR.preprocess import click_probs

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  
  flags.DEFINE_string(  'dataset_name', 'MSLR-WEB10k', 
                        'name of dataset: "MSLR-WEB10k" or "Webscope_C14_Set1"')
  flags.DEFINE_string(  'datasets_info_path', '/Users/aliv/eclipse-workspace/myModules/trust_bias/preprocess/datasets_info.json', 
                        'path to the datasets info file.')
  flags.DEFINE_integer( 'data_fold', 0, 
                        'data fold number')

  flags.DEFINE_string(  'click_probs_pickle_path', '/Users/aliv/MySpace/_DataSets/LTR/Microsoft/MSLR-WEB10K/Fold1/trust_1_top50.lgbm_20_3._.pkl',
                        'path to the pickle file containing click probs of docs.')
  
  flags.DEFINE_string(  'click_count', '2**19',
                        'the default value for topk when it is not specified in the click policy file.')

  flags.DEFINE_integer( 'pickle_dump_protocol', 2,
                        'protocol number for pickle dump.')


def _get_cleaned(vec, rank):
  cleaned = np.zeros_like(vec)
  cleaned[rank] = vec
  return cleaned

class ClicksSplit:
  def __init__(self, all_clicks_dict, split_name):
    self.name = split_name
    self.clicks = all_clicks_dict[ClicksSplit.clicks_key(split_name)]
    self.shown = all_clicks_dict[ClicksSplit.shown_key(split_name)]
    self.doclist_ranges = all_clicks_dict[ClicksSplit.ranges_key(split_name)]
    self.ranks = all_clicks_dict[ClicksSplit.ranks_key(split_name)]
    

  def query_range(self, query_index):
    s_i = self.doclist_ranges[query_index]
    e_i = self.doclist_ranges[query_index+1]
    return s_i, e_i
  
  
  def _query_click_rates(self, query_index):
    s_i, e_i = self.query_range(query_index)
    clicks = self.clicks[s_i:e_i]
    shown = self.shown[s_i:e_i]
    
    rates, shown_len = ClicksSplit.get_click_rates(clicks, shown)
    
    return rates, clicks, shown, self.ranks[s_i:e_i], shown_len
  
#   def list_click_rates(self, qids):
#     rates = []
#     ranks = []
#     ranges = [0]
#     for qid in qids:
#       rate, rank, range = self._query_click_rates(qid)
#       rates.append(rate)
#       ranks.append(rank)
#       ranges.append(range)
#       
#     return np.concatenate(rates, axis=0), np.concatenate(ranks, axis=0), np.cumsum(np.array(ranges), axis=0)
  
  def clean_and_sort(self):
    click_rates = []
    click_counts = []
    click_showns = []
    doclist_ranges = [0]
    argsorted = []
    
    for qid in range(self.doclist_ranges.shape[0] - 1):
      rate, click, shown, rank, shown_len = self._query_click_rates(qid)
      if shown_len > 0:
        click_rates.append(_get_cleaned(rate, rank)[:shown_len])
        click_counts.append(_get_cleaned(click, rank)[:shown_len])
        click_showns.append(_get_cleaned(shown, rank)[:shown_len])
        doclist_ranges.append(shown_len)
        irank = np.argsort(rank)
        argsorted.append(self.doclist_ranges[qid] + irank[:shown_len])
      
    self.click_rates = np.concatenate(click_rates, axis=0)
    self.click_counts = np.concatenate(click_counts, axis=0)
    self.click_showns = np.concatenate(click_showns, axis=0)
    
    self.argsorted = np.concatenate(argsorted, axis=0)
    self.new_doclist_range = np.cumsum(np.array(doclist_ranges), axis=0)
    
  @staticmethod
  def get_click_rates(clicks, shown):
    rates = -np.ones_like(clicks, dtype=np.float64)
    non_zero_index = shown != 0.
    rates[non_zero_index] = (1. * clicks[non_zero_index]) / shown[non_zero_index]
    return rates, len(shown[non_zero_index])
  
  @staticmethod
  def clicks_key(name):
    return name+'_clicks'
  @staticmethod
  def shown_key(name):
    return name+'_shown'
  @staticmethod
  def ranges_key(name):
    return name+'_doclist_ranges'
  @staticmethod
  def ranks_key(name):
    return name+'_ranks'
  
class Clicks:
  def __init__(self, clicks_pickle_path):
    with open(clicks_pickle_path, 'rb') as f:
      all_clicks = pickle.load(f, encoding='latin1')
      
    self.train = ClicksSplit(all_clicks, 'train')
    self.valid = ClicksSplit(all_clicks, 'valid')
    self.test = ClicksSplit(all_clicks, 'test')
    self.clicks_split_by_name = {'train': self.train, 'valid': self.valid, 'test': self.test}
    
def load_clicks_pickle(clicks_pickle_path):
  return Clicks(clicks_pickle_path)



def simulate_clicks(data_fold_split, click_probs, click_count):
  cnt = 0
  clicks_per_doc = np.zeros_like(data_fold_split.label_vector, dtype=np.int64)
  shown_per_doc = np.zeros_like(data_fold_split.label_vector, dtype=np.int64)
  
  while cnt < click_count:
    qid = np.random.randint(0, data_fold_split.num_queries())
    s_i, e_i = data_fold_split.query_range(qid)
    probs_snapshot = np.array(click_probs[s_i:e_i])
    probs_snapshot[probs_snapshot == -1.] = 0.
    clicks_snapshot = np.random.binomial(1, probs_snapshot)
    s = np.sum(clicks_snapshot)
#     if s > 0:
    shown_snapshot = np.zeros_like(probs_snapshot, dtype=np.int64)
    shown_snapshot[click_probs[s_i:e_i] >= 0.] = 1
    cnt += s
    clicks_per_doc[s_i:e_i] += clicks_snapshot
    shown_per_doc[s_i:e_i] += shown_snapshot
        
  return clicks_per_doc, shown_per_doc

def main(args):
  s_time = time.time()
  data = dataset.get_dataset_from_json_info(
                    FLAGS.dataset_name,
                    FLAGS.datasets_info_path,
                  ).get_data_folds()[FLAGS.data_fold]
  data.read_data()
  print('read data in {} seconds.'.format(time.time() - s_time))
   
#   with open(FLAGS.click_probs_pickle_path, 'rb') as f:
#     all_click_probs = pickle.load(f)
  all_click_probs = click_probs.load_click_probs_pickle(FLAGS.click_probs_pickle_path)
  
  click_count = int(eval(FLAGS.click_count))
  
  all_clicks = {}
  print('starting to simulate {} clicks.'.format(click_count))
  for dfs in [data.train, data.valid, data.test]:
    s_time = time.time()
    click_probs_split = all_click_probs.click_probss_split_by_name[dfs.name]
    
    dfs_count = click_count * (dfs.num_queries() * 1. / data.train.num_queries())
    clicks_per_doc, shown_per_doc = simulate_clicks(dfs, click_probs_split.probs, dfs_count)
    
    all_clicks[ClicksSplit.clicks_key(dfs.name)] = clicks_per_doc
    all_clicks[ClicksSplit.shown_key(dfs.name)] = shown_per_doc
    all_clicks[ClicksSplit.ranges_key(dfs.name)] = dfs.doclist_ranges
    all_clicks[ClicksSplit.ranks_key(dfs.name)] = click_probs_split.ranks
    
    e_time = time.time()
    print('simulated {} clicks for {} dataset in {} seconds.'.format(dfs_count, dfs.name, e_time - s_time))
    
  dirname = os.path.dirname(FLAGS.click_probs_pickle_path)
  basename = os.path.basename(FLAGS.click_probs_pickle_path)
  
  with open(os.path.join(dirname, '{}.{}'.format(click_count, basename)), 'wb') as f:
    pickle.dump(all_clicks, f, protocol=FLAGS.pickle_dump_protocol)

if __name__ == '__main__':
  app.run(main)


def test():
  clicks_split = load_clicks_pickle('/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/262144.trust_1_top5.lgbm_20_3.set1.pkl')
  clicks_split2 = load_clicks_pickle('/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/262144.ideal_top20.lgbm_20_3.set1.pkl')
  a=clicks_split.train
  a2=clicks_split2.train
  a.clean_and_sort()
  a2.clean_and_sort()
  b = a.ranks[a.argsorted]
  b2 = a2.ranks[a2.argsorted]
  print(b[:100])
  print(b2[:100])