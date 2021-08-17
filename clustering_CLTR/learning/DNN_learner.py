'''
Created on 9 Jun 2020

@author: aliv
'''

from .base import BaseLearner
from . import tf_losses
import numpy as np
import tensorflow as tf
import logging
logger = logging.getLogger('learner')

def sigmoid_prob(logits):
  return 1.*tf.sigmoid(1.*(logits - tf.reduce_mean(logits, -1, keepdims=True)))

def min_max_prob(logits):
  e = tf.exp(logits)
  e = e - tf.reduce_min(e,-1,keepdims=True)
  return 1. * e / tf.reduce_max(e, -1, keepdims=True)


class data_util():
  def __init__(self, data, topk = None):
    self.feature_matrix = data.feature_matrix
    self.label_vector = data.label_vector
    self.doclist_ranges = data.doclist_ranges
    
    self.max_ranklist_size = topk if topk is not None else np.max(np.diff(self.doclist_ranges))
    
    self.last_used_batch_index = -1
    self.samples_size = self.doclist_ranges.shape[0] - 1
  
  
  def get_random_indexes(self, batch_size):
    if self.last_used_batch_index == -1:
      self.epochs = -1
      self.last_used_batch_index = self.samples_size
    
    if self.last_used_batch_index >= self.samples_size:
      self.permuted_indices = np.random.permutation(self.samples_size)
      self.last_used_batch_index = 0
      self.epochs += 1
      
    start = self.last_used_batch_index
    end = self.last_used_batch_index + batch_size
    if end > self.samples_size:
      end = self.samples_size
    self.last_used_batch_index = end
    
    return self.permuted_indices[start:end]
  
  
  def load_batch(self, qids):
    feature_matrix = []
#     click_rates = []
    padded_labels = []
    padding_mask = []
    if qids is None:
      qids = range(self.samples_size)
    for qid in qids:
      s_i = self.doclist_ranges[qid]
      e_i = self.doclist_ranges[qid+1]
      if e_i - s_i > self.max_ranklist_size:
        e_i = s_i + self.max_ranklist_size
      feature_matrix.append(self.feature_matrix[s_i:e_i, :])
      feature_matrix.append(np.zeros([self.max_ranklist_size - e_i + s_i, self.feature_matrix.shape[1]], dtype=np.float64))
#       click_rates.append(self.click_rates[s_i:e_i])
#       click_rates.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      padded_labels.append(self.label_vector[s_i:e_i])
      padded_labels.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      padding_mask.append(np.ones([e_i - s_i], dtype=np.float64))
      padding_mask.append(np.zeros([self.max_ranklist_size - e_i + s_i], dtype=np.float64))
      
    return np.concatenate(feature_matrix, axis=0), np.concatenate(padded_labels, axis=0), np.concatenate(padding_mask, axis=0)
    
  
class DNNLearner(BaseLearner):
  def __init__(self, 
               batch_size,
               loss_function_str,
               max_train_iterations,
               layers_size = [512, 256, 128], 
               drop_out_probs = [0.0,0.1,0.1],
               learning_rate = 4e-3,
               max_gradient_norm = 10, 
               optimizer = 'adagrad', 
               embed_size = 501,
               max_ranklist_size = 50
               ):

    super(DNNLearner, self).__init__()
    self._drop_out_probs = np.zeros(len(layers_size), dtype=np.float64)
    for i in range(min(len(layers_size),len(drop_out_probs))):
      self._drop_out_probs[i] = drop_out_probs[i]

    self._loss_function_str = loss_function_str
    self._hidden_layer_sizes = list(map(int, layers_size))
    self._max_gradient_norm = max_gradient_norm
    
    self.batch_size = batch_size
    self.max_ranklist_size = max_ranklist_size
    self.max_train_iterations = max_train_iterations
    
    self._embed_size = embed_size
    self.build_network()
    
    if optimizer == 'adagrad':
#       self._optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
      self._optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'sgd':
      self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
      self._optimizer = tf.keras.optimizers.get(optimizer)
        
    if self._loss_function_str.startswith('sigmoid'):
      self._loss_class = tf_losses.SigmoidLoss(self.max_ranklist_size)
    if self._loss_function_str == 'softmax':
      self._loss_class = tf_losses.SoftmaxLoss(self.max_ranklist_size)
    elif self._loss_function_str == 'lambdaloss':
      self._loss_class = tf_losses.LambdaLoss(self.max_ranklist_size)
      
    
#     self.network_model.compile(optimizer=opt, loss=self._loss_class.loss_fn())
    self.global_step = 0
      

  def build_network(self):
    current_size = self._embed_size
    self.ltf_w = []
    self.ltf_b = []
    
    for layer in range(len(self._hidden_layer_sizes)):
      fan_in = current_size
      fan_out = self._hidden_layer_sizes[layer]
#       glorot_uniform_initializer as is default in tf.get_variable()
      r = np.sqrt(6.0/(fan_in+fan_out))
#       tf_w = tf.Variable(tf.random_normal([current_size, self._hidden_layer_sizes[layer]], stddev=0.1), name='rel/w_{}'.format(layer))
      self.ltf_w.append(tf.Variable(tf.random.uniform([fan_in, fan_out], minval=-r, maxval=r, dtype=tf.float64), name='trust/w_{}'.format(layer)))
      self.ltf_b.append(tf.Variable(tf.constant(0.1, shape=[fan_out], dtype=tf.float64), name='trust/b_{}'.format(layer)))

      current_size = self._hidden_layer_sizes[layer]
    
    
    # Output layer
  
    fan_in = self._hidden_layer_sizes[-1]
    fan_out = 1
    r = np.sqrt(6.0/(fan_in+fan_out))
    self.ltf_w.append(tf.Variable(tf.random.uniform([fan_in, fan_out], minval=-r, maxval=r, dtype=tf.float64), name='rel/w_last'))
    self.ltf_b.append(tf.Variable(tf.constant(0.1, shape=[fan_out], dtype=tf.float64), name='rel/b_{}'.format('last')))


  @tf.function
  def network(self,
              feature_matrix,
              training):
    
    tf_output = feature_matrix
    
    for layer in range(len(self._hidden_layer_sizes)):
      
      tf_w = self.ltf_w[layer]
      tf_b = self.ltf_b[layer]
      # x.w+b
      tf_output_tmp = tf.nn.bias_add(tf.matmul(tf_output, tf_w, name='trust/mul_{}'.format(layer)), tf_b, name='trust/affine_{}'.format(layer))
      # activation: elu
      tf_output = tf.nn.elu(tf_output_tmp, name='trust/elu_{}'.format(layer))
      
      if self._drop_out_probs[layer] > 0.0 and training:
        tf_output = tf.nn.dropout(tf_output, rate=self._drop_out_probs[layer], name = 'rel/drop_out_{}'.format(layer))

    
    # Output layer
    tf_w = self.ltf_w[-1]
    tf_b = self.ltf_b[-1]
    
    tf_output = tf.nn.bias_add(tf.matmul(tf_output, tf_w), tf_b, name='rel/affine_last')
    
    return tf_output


  def train_on_batch(self, mDataset):
    indexes = mDataset.get_random_indexes(self.batch_size)
    feature_matrix, labels_, padding_mask = mDataset.load_batch(indexes)
    
    labels = np.reshape(labels_, [-1, self.max_ranklist_size])
    mask = np.reshape(padding_mask, [-1, self.max_ranklist_size])
    
    self._loss_class.set_mask(mask)

    with tf.GradientTape() as tape:
      loss = self._loss_class.loss_fn()(y_true=mask * labels,
                                        y_pred=self.network(feature_matrix, training=True))
    
#     self._optimizer.minimize(loss, lambda: self.network_model.trainable_weights)

      params = self.ltf_w + self.ltf_b
#       print(params)
      gradients = tape.gradient(loss, params)
      if self._max_gradient_norm > 0:
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
        self._optimizer.apply_gradients(zip(clipped_gradients, params))
      else:
        self._optimizer.apply_gradients(zip(self.gradients, params))
          
          
    self.global_step += 1
    return loss
    
  def test_on_batch(self, mDataset, seq_index):
    if seq_index >= mDataset.samples_size:
      raise Exception('sequential batch start index exceeds total number of samples!')
  
    start = seq_index
    seq_index += self.batch_size
    end = seq_index if seq_index <= mDataset.samples_size else mDataset.samples_size
    
    
    indexes = np.array(list(range(start,end)))
    feature_matrix, click_rates, padding_mask = mDataset.load_batch(indexes)
    
    labels = np.reshape(click_rates, [-1, self.max_ranklist_size])
    mask = np.reshape(padding_mask, [-1, self.max_ranklist_size])
    
    self._loss_class.set_mask(mask)
    loss = self._loss_class.loss_fn()(y_true=mask * labels,
                                          y_pred=self.network(feature_matrix, training=False))
    return loss

  def predict(self, testset):

    it = 0
    
    feature_matrix, labels = testset.feature_matrix, testset.label_vector
    l_out = []

    while it < feature_matrix.shape[0]:
      end = it + (self.batch_size  * self.max_ranklist_size)
      if end > feature_matrix.shape[0]:
        end = feature_matrix.shape[0]
      input = feature_matrix[it:end,:]
      out = self.network(input, training=False)
      
      out = tf.reshape(out,[-1])
      l_out.append(out)
      it = end
      
    bin_labels = np.zeros_like(labels, dtype=np.float64)
    bin_labels[labels>2] = 1.
    return np.concatenate(l_out, 0)
  
  
  def load_saved_model(self, path):
    pass
    
  def train(self, trainset, validset):
    
    mDataset = data_util(trainset, self.max_ranklist_size)
#     valid_dataset = data_util(validset, self.max_ranklist_size)
    for iter in range(self.max_train_iterations):
      self.train_on_batch(mDataset)
    

