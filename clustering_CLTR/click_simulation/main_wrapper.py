'''
Created on 5 Jun 2020

@author: aliv
'''
import json
import time
import os
from .pbm_click_simulation import PBMClickSimulation
from .dcm_click_simulation import DCMClickSimulation
from .ubm_click_simulation import UBMClickSimulation
from .dbn_click_simulation import DBNClickSimulation

def _load_click_policies(click_policy_path, default_topk):
  with open(click_policy_path) as f:
    click_policies_dict = json.load(f)
  click_policies = {}
  for name in click_policies_dict:
    if name == 'comment':
      continue
    click_policies[name] = {'pbm':PBMClickSimulation,
                            'dcm':DCMClickSimulation,
                            'ubm':UBMClickSimulation,
                            'dbn':DBNClickSimulation
                            }[click_policies_dict[name]['type']](
                              model_name = name, 
                              json_description = click_policies_dict[name], 
                              default_topk = default_topk)
  return click_policies
    
    
def simulate_clicks(data,
                    click_count,
                    click_policy_path,
                    all_ranks_pickle,
                    output_dir,
                    default_topk):
  
  click_policies = _load_click_policies(click_policy_path, default_topk)
  click_count = int(click_count)
  for name in click_policies:
    start_time = time.time()
    click_policies[name].simulate_clicks(data.train, 
                                         all_ranks_pickle['train'], 
                                         click_count, 
                                         os.path.join(output_dir, 'clicks.train.{}.{}.pkl'.format(click_count, name)))
    click_policies[name].simulate_clicks(data.valid, 
                                         all_ranks_pickle['valid'], 
                                         click_count * (data.valid.num_queries() * 1. / data.train.num_queries()), 
                                         os.path.join(output_dir, 'clicks.valid.{}.{}.pkl'.format(click_count, name)))
    print('dumped {}. took {} seconds.'.format(name, time.time() - start_time))