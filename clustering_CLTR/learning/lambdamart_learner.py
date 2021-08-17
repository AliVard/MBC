'''
Created on 9 Jun 2020

@author: aliv
'''

from .base import BaseLearner
import numpy as np
import lightgbm as lgb

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

class LambdamartLearner(BaseLearner):
  def __init__(self, early_stopping_rounds, eval_at, loss_function_str = None):
    super(LambdamartLearner, self).__init__()
#     self.gbm = lgb.LGBMRegressor()
    self.gbm = lgb.LGBMRanker(learning_rate=0.05, n_estimators=300)
#     learning_rate=0.05, n_estimators=300 -> nDCG@10:0.6812069139318636
#     learning_rate=0.1, n_estimators=100 -> nDCG@10:0.6737754428776237
    self.early_stopping_rounds = early_stopping_rounds
    self.eval_at = eval_at
  
  def load_saved_model(self, path):
    pass
    

  def train(self, trainset, validset):
    #   true_y[true_y>0.90] = 1
#   true_y[true_y<0.01] = 0
#     trainset.label_vector[(trainset.label_vector>0.9) & (trainset.label_vector<1)] = 1
#     trainset.label_vector[trainset.label_vector<0.1] = 0
    trainset.label_vector *= 4.
#     validset.label_vector[(validset.label_vector>0.9) & (validset.label_vector<1)] = 1
#     validset.label_vector[validset.label_vector<0.1] = 0
    validset.label_vector *= 4.
    self.gbm.fit(trainset.feature_matrix, trainset.label_vector, 
          group=np.diff(trainset.doclist_ranges), 
          eval_set=[(validset.feature_matrix, validset.label_vector)],
          eval_group=[np.diff(validset.doclist_ranges)], 
          eval_at=self.eval_at, 
          early_stopping_rounds=self.early_stopping_rounds, 
          verbose=False)
  
  def predict(self, testset):
    booster = self.gbm.booster_
    return booster.predict(testset.feature_matrix)
    
#   def test(self, testset):
#     y_pred = self.predict(testset)
# #     lv = np.zeros_like(testset.label_vector)
# #     lv[testset.label_vector>2] = 1
#     lv = testset.label_vector
#     metric = LTRMetrics(lv,np.diff(testset.doclist_ranges),y_pred)
#     return metric.NDCG(10)

