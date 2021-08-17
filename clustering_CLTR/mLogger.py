'''
Created on 16 Oct 2020

@author: aliv
'''
import logging

mLoggers = {
  'correction':logging.getLogger('correction'), 
  'time':logging.getLogger('time'), 
  'learner':logging.getLogger('learner'), 
  'data':logging.getLogger('data'), 
  'clustering_correction':logging.getLogger('clustering_correction'),
  'enhance':logging.getLogger('enhance'),
  'simulation':logging.getLogger('simulation'),
  'metric':logging.getLogger('metric')}

def enable_debugs(debugs_to_show):
  for debug_to_show in debugs_to_show:
    if debug_to_show in mLoggers:
      mLoggers[debug_to_show].setLevel(logging.DEBUG)