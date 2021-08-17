'''
Created on Fri Mar  6  2020

@author: aliv
'''

import os
import re

def get_files(data_dir):
  if os.path.isfile(data_dir):
    data_files = [os.path.basename(data_dir)]
    data_dir = os.path.dirname(data_dir)
  else:
    if os.path.exists(data_dir):
      data_files = os.listdir(data_dir)
    else:
      pattern = os.path.basename(data_dir).replace('.','\.').replace('*','.*')
      data_dir = os.path.dirname(data_dir)
      data_files = [f for f in os.listdir(data_dir) if re.match("(?:" + pattern + r")\Z", f)]

    
    data_files.sort()
    
  return data_dir, data_files