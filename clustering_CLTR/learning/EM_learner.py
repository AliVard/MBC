'''
Created on 25 Jun 2020

@author: aliv
'''
import numpy as np
from .base import BaseLearner, dcg_binary_labels
from .DNN_learner import DNNLearner
from .lambdamart_learner import LambdamartLearner
import logging
logger = logging.getLogger('learner')

class EMLearner(object):
  def __init__(self, learner, correction):
    self.learner = learner
    self.correction = correction
    
  def EM_train(self, trainset, validset, iterations, logits_to_probs_fn, test_learner = None, testset = None, binary_rel = False):
#     logging.getLogger('mixture').log(logging.DEBUG,'--------------')
    previous_alphas = []
    previous_betas = []
    for iteration in range(iterations):
      self.learner.train(trainset, validset)
      if testset is not None and (iteration % 5 == 0 or iteration < 5 or (iteration == iterations - 1)):
        if test_learner is not None:
          learner = test_learner
          learner.train(trainset, validset)
        else:
          learner = self.learner
        ndcg = learner.test(testset, binary_rel=binary_rel)
        logger.debug('iteration:{}, test ndcg@10:{}'.format(iteration, ndcg))
#         if iteration == 30:
#           ndcg_30 = ndcg
#           alpha_30 = np.copy(self.correction.alpha)
#           beta_30 = np.copy(self.correction.beta)
        
      logger.debug('alpha:{}'.format(list(self.correction.alpha)))
      logger.debug('beta:{}'.format(list(self.correction.beta)))
      logger.debug('--------------')
        
      y_pred = self.learner.predict(trainset)
      self.correction.update(y_pred, logits_to_probs_fn)
      if self.correction.alpha is not None:
        previous_alphas.append(np.copy(self.correction.alpha))
        previous_betas.append(np.copy(self.correction.beta))
      else:
        self.correction.alpha = previous_alphas[-10]
        self.correction.beta = previous_betas[-10]
        break
      
#       trainset.label_vector = self.correction.gamma
#       valid_y_pred = self.learner.predict(validset)
#       self.correction.get_validation_gamma(valid_y_pred, logits_to_probs_fn)
#       validset.label_vector = self.correction.valid_gamma
      
      
      self.correction.correct()
      trainset.label_vector = self.correction.corrected_clicks['train']
      validset.label_vector = self.correction.corrected_clicks['valid']
    
#     if ndcg[0] < ndcg_30[0]:
#       self.correction.alpha = np.copy(alpha_30)
#       self.correction.beta = np.copy(beta_30)
#       ndcg = ndcg_30
    return {'dcg':ndcg, 'alpha':self.correction.alpha, 'beta':self.correction.beta}
    
    
    
  def EM_train_validation(self, trainset, validset, iterations, logits_to_probs_fn, testset = None):
#     logging.getLogger('mixture').log(logging.DEBUG,'--------------')
    best_alpha = None
    best_beta = None
    best_dcg = 0.
    ndcg_list = []
    for _ in range(iterations):
#       self.correction.init_EM()
#       self.learner = DNNLearner()
      self.learner.train(trainset, validset)
      if testset is not None:
        valid_dcg = self.learner.test(validset, dcg_binary_labels)
        dcg = self.learner.test(testset, dcg_binary_labels)
        ndcg = self.learner.test(testset)
        if valid_dcg > best_dcg:
          best_alpha = self.correction.alpha
          best_beta = self.correction.beta
          best_dcg = valid_dcg
        logger.debug('validation dcg@10:{}, test dcg@10:{}, test ndcg@10:{}'.format(valid_dcg, dcg, ndcg))
        logger.debug('positive:{}'.format(list(self.correction.alpha[:10] + self.correction.beta[:10])))
        logger.debug('negative:{}'.format(list(self.correction.beta[:10])))
        logger.debug('--------------')
        ndcg_list.append(valid_dcg)
        if len(ndcg_list) > 5:
          going_bad = True
          for backwards_i in range(-5,0):
            going_bad &= ndcg_list[backwards_i] < ndcg_list[backwards_i - 1]
          if going_bad:
            break
          if len(ndcg_list) > 15:
            if ndcg_list[-15] > max(ndcg_list[-14:-1]):
              break
        
      y_pred = self.learner.predict(trainset)
#       rel_probs = logits_to_probs_fn(y_pred)
      self.correction.update(y_pred, logits_to_probs_fn)
#       trainset.label_vector = self.correction.gamma
      trainset.label_vector = self.correction.corrected_clicks['train']
      validset.label_vector = self.correction.corrected_clicks['valid']
      
    logger.debug('best: dcg:{}, alpha:{}, beta:{}'.format(best_dcg, best_alpha[:3], best_beta[:3]))
    if best_alpha is not None:
      self.correction.alpha = best_alpha
      self.correction.beta = best_beta
      return {'dcg':best_dcg, 'alpha':best_alpha, 'beta':best_beta}
    else:
      return None