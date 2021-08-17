'''
Created on 5 Jun 2020

@author: aliv
'''

from absl import app
from absl import flags
import os
import sys

import numpy as np

SYS_PATH_APPEND_DEPTH = 2
SYS_PATH_APPEND = os.path.abspath(__file__)
for _ in range(SYS_PATH_APPEND_DEPTH):
  SYS_PATH_APPEND = os.path.dirname(SYS_PATH_APPEND)
sys.path.append(SYS_PATH_APPEND)

from clustering_CLTR.dataset import get_dataset_from_json_info
from clustering_CLTR.click_simulation.main_wrapper import simulate_clicks
from clustering_CLTR.learning.main_wrapper import train_and_test
from clustering_CLTR.click_simulation.oracle_weights import get_oracle_weights
from clustering_CLTR import mLogger
from clustering_CLTR.correction.compare_corrections import compare_with_gold, plot_corrected

from clustering_CLTR.correction.pbm_affine_correction import PBMAffineCorrection
from clustering_CLTR.correction.pbm_clustering_correction import PBMClusteringCorrection

import pickle
import json
from clustering_CLTR.ctr_dist import plot_clicks, plot_two_clicks, plot_multiple_separate_clicks

MY_CODE_DIRCTORY = '/Users/aliv/eclipse-workspace/myModules/clustering_CLTR'
# MY_CODE_DIRCTORY = 'clustering_CLTR'

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_string(  'datasets_info_path', '{}/preprocess/datasets_info.json'.format(MY_CODE_DIRCTORY), 
                        'path to the datasets info file.')
  flags.DEFINE_string(  'model_dir', '/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/lgbm_20_3.txt', 
                        'with LambdaMART model directory for saving trained model.')
  flags.DEFINE_string(  'click_policy_path', '{}/preprocess/click_policy.json'.format(MY_CODE_DIRCTORY), 
                        'path to the file containing the policy for simulating clicks.')
#   flags.DEFINE_string(  'ranks_pickle_path', '/Users/aliv/MySpace/_DataSets/LTR/Microsoft/MSLR-WEB30K/Fold1/ranks.lgbm_20_1._.pkl', 'path to the pickle file containing ranks of docs.')
  flags.DEFINE_string(  'ranks_pickle_path', '/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/ranks.lgbm_20_3.set1.pkl', 'path to the pickle file containing ranks of docs.')
#   flags.DEFINE_string(  'train_clicks_pickle_path', '/Users/aliv/MySpace/_DataSets/LTR/Microsoft/MSLR-WEB30K/clicks/clicks.train.2097154.trust_1_top50.pkl','path to the pickle file containing ranks of docs.')
  flags.DEFINE_string(  'train_clicks_pickle_path', '/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/clicks.train.20000.ubm_1_top10.pkl','path to the pickle file containing ranks of docs.')
  flags.DEFINE_bool(    'binary_rel', False, '')
  
  flags.DEFINE_string(  'module', 'simulate_clicks', 'which module to run? "ctr_dist", "simulate_clicks", "oracle_weights" or "train_and_test", "compare"')
  
  flags.DEFINE_string(  'dataset_name', 'Webscope_C14_Set1', 'name of dataset: "MSLR-WEB30k" or "Webscope_C14_Set1"')
#   flags.DEFINE_string(  'dataset_name', 'MSLR-WEB30k', 'name of dataset: "MSLR-WEB30k" or "Webscope_C14_Set1"')
  flags.DEFINE_integer( 'data_fold', 0, 'data fold number')
  
  flags.DEFINE_string(  'click_count', '2e4', 'number of clicks to simulate.')
  
  flags.DEFINE_integer( 'topk', 10, 'cutting top k number of items')
  
  flags.DEFINE_string(  'learning_algorithm', 'lambdamart', '"lambdamart" or "DNN" Add "EM_" at the beginning to infer the params using EM.')  
  flags.DEFINE_string(  'correction_method', 'pbm_soft_clustering', '"full_info", "no_correction", "pbm_soft_clustering", "pbm_hard_clustering", or "pbm_affine"')  
  flags.DEFINE_string(  'regression_function', 'DNN', '"lambdamart" or "DNN" for regression part of the regression-based EM.')  
  flags.DEFINE_string(  'mixture', 'gaussian', '"binomial" or "gaussian"')
  
  
  flags.DEFINE_integer( 'n_components', 2, 'number of mixture model components')
  flags.DEFINE_integer( 'enhance_data', 5, 'multiply number of sessions by this amount to enhance the CTR data')
  flags.DEFINE_string(  'enhance_distribution', 'binomial', 'distribution used to enhance the CTR data. "binomial", "uniform"')
  flags.DEFINE_integer( 'default_topk', 1000,
                        'the default value for topk when it is not specified in the click policy file for simulating clicks.')
  
#   5-> nDCG@10:0.6895176611403554
#   10-> nDCG@10:0.6897458416804101
  flags.DEFINE_string(  'output_path', 'output.txt',
                        'path to the output file containing the summary of results.')
  flags.DEFINE_string(  'slurm_job_id', '0_0',
                        'slurm job id.')
  
  flags.DEFINE_string(  'DNN_loss_function_str', 'sigmoid_zeromean', '"sigmoid", "sigmoid_zeromean", "softmax" or "lambdaloss"')
  flags.DEFINE_integer( 'DNN_max_train_iterations', 5, 'number of iterations for DNN train batchs.')
  flags.DEFINE_integer( 'DNN_batch_size', 8, 'batch size for DNN train.')
  flags.DEFINE_string(  'DNN_learning_rate', '4e-3', '')
  flags.DEFINE_integer( 'DNN_embed_size', 501, 'embedding size.')
  
  flags.DEFINE_integer(  'lambdamart_early_stopping_rounds', 10000, '')
  flags.DEFINE_string(  'lambdamart_eval_at', '[10]', '')
  
  flags.DEFINE_string(  'regression_loss_function_str', 'sigmoid', '"sigmoid", "sigmoid_zeromean", "softmax" or "lambdaloss"')
  flags.DEFINE_integer( 'regression_DNN_max_train_iterations', 16, 'number of iterations for DNN train batchs.')
  flags.DEFINE_integer( 'regression_DNN_batch_size', 16, 'batch size for DNN train.')
  flags.DEFINE_string(  'regression_DNN_learning_rate', '4e-3', '')
  flags.DEFINE_integer( 'regression_DNN_embed_size', 501, 'embedding size.')
  
  flags.DEFINE_string( 'EM_max_iterations', '100', '')
  flags.DEFINE_integer( 'regression_lambdamart_early_stopping_rounds', 10000, '')
  flags.DEFINE_string(  'regression_lambdamart_eval_at', '[10]', '')
  
  flags.DEFINE_string(  'debugs', 'correction,data,learner', 'comma-separated list of debugs to show: correction,learner,data,clustering_correction,enhance')
  

def _get_pbm_affine_json():
  filename = os.path.basename(FLAGS.train_clicks_pickle_path)
  model_name = filename.split('.')[-2]
  with open(FLAGS.click_policy_path) as f:
    clicks_info = json.load(f)
  if model_name in clicks_info:
    return clicks_info[model_name]
  else:
    return clicks_info['comment'][model_name]

def data_stats(data_fold):
  q, d = data_fold.num_queries(), data_fold.num_docs()
  mLogger.mLoggers['data'].debug('queries:{}, docs:{}'.format(q,d))
  return q, d

oracle_rel_weights = {
  'Webscope_C14_Set1':[[0.6971869829012687, 0.3028130170987314], [0.8146718146718147, 0.18532818532818532], [0.8561399989971419, 0.14386000100285815], [0.8858747430176002, 0.11412525698239984], [0.8919921777064634, 0.10800782229353657], [0.9065336208193351, 0.09346637918066489], [0.9162613448327734, 0.08373865516722659], [0.9218272075414933, 0.07817279245850674], [0.9301509301509301, 0.06984906984906986], [0.9345133630847917, 0.06548663691520834], [0.9364689364689365, 0.06353106353106353], [0.9444918016346587, 0.055508198365341224], [0.9492553778268064, 0.0507446221731936], [0.9512109512109512, 0.048789048789048786], [0.9571779571779572, 0.04282204282204282], [0.9576793862508148, 0.04232061374918518], [0.9602366745223888, 0.03976332547761119], [0.9587323873038158, 0.041267612696184126], [0.9648498219926791, 0.03515017800732086], [0.963395677681392, 0.03660432231860803], [0.967858396429825, 0.032141603570175], [0.966805395376824, 0.033194604623176055], [0.9688612545755403, 0.03113874542445971], [0.9727222584365441, 0.02727774156345585], [0.9715188286616858, 0.028481171338314196], [0.9735245449531164, 0.026475455046883617], [0.9754299754299754, 0.02457002457002457], [0.9768339768339769, 0.023166023166023165], [0.9793411221982651, 0.020658877801734945], [0.9783884069598355, 0.021611593040164467], [0.9797422654565512, 0.02025773454344883], [0.9810459810459811, 0.018954018954018954], [0.9825001253572682, 0.017499874642731787], [0.985308128165271, 0.014691871834728977], [0.9852078423506995, 0.014792157649300506], [0.9867121295692725, 0.013287870430727574], [0.9869628441057012, 0.013037155894298751], [0.9857092714235571, 0.014290728576442863], [0.9879657022514166, 0.012034297748583464], [0.9888181316752745, 0.011181868324725468], [0.9900717043574186, 0.009928295642581356], [0.9908238479667051, 0.009176152033294891], [0.9914255628541343, 0.008574437145865718], [0.992127563556135, 0.007872436443865016], [0.9920272777415634, 0.007972722258436545], [0.9931805646091361, 0.006819435390863962], [0.9925788497217068, 0.0074211502782931356], [0.993631850774708, 0.006368149225292083], [0.9933309933309933, 0.006669006669006669], [0.9945845660131375, 0.0054154339868625586]],
  'MSLR-WEB30k':[[0.8510413362934771, 0.1489586637065229], [0.9091341579448144, 0.09086584205518554], [0.9255735278570674, 0.07442647214293266], [0.9404799661697854, 0.05952003383021461], [0.9469288508298974, 0.05307114917010255], [0.9496775557669944, 0.0503224442330056], [0.953272016069352, 0.04672798393064806], [0.9585579871022307, 0.04144201289776932], [0.9598266201501215, 0.04017337984987842], [0.9628396236388624, 0.03716037636113754], [0.9636325192937942, 0.03636748070620573], [0.9630510624801776, 0.036948937519822395], [0.9633153610318216, 0.03668463896817845], [0.9643196955280685, 0.0356803044719315], [0.9638439581351094, 0.03615604186489058], [0.9670684004651654, 0.03293159953483455], [0.9708742996088382, 0.029125700391161857], [0.968918490326673, 0.03108150967332699], [0.970662860767523, 0.029337139232477005], [0.9714557564224549, 0.028544243577545196], [0.9703457025055503, 0.02965429749444973], [0.9718786341050851, 0.028121365894914895], [0.9712443175811396, 0.028755682418860343], [0.9717729146844275, 0.02822708531557247], [0.9725658103393593, 0.02743418966064066], [0.974627339042182, 0.02537266095781795], [0.9754730944074427, 0.024526905592557352], [0.9740458822285654, 0.025954117771434613], [0.974151601649223, 0.025848398350777037], [0.9753145152764563, 0.024685484723543715], [0.9755788138281002, 0.024421186171899777], [0.9750502167248123, 0.024949783275187654], [0.9747330584628396, 0.025266941537160376], [0.9748387778834972, 0.0251612221165028], [0.977376043979279, 0.022623956020721005], [0.9764774289036896, 0.023522571096310393], [0.9774817633999365, 0.02251823660006343], [0.9768474468759911, 0.02315255312400888], [0.9767417274553335, 0.023258272544666454], [0.9785918173168411, 0.021408182683158895], [0.9764245691933608, 0.02357543080663918], [0.9786446770271698, 0.021355322972830108], [0.9797547309440744, 0.020245269055925573], [0.9780103605032244, 0.021989639496775557], [0.978380378475526, 0.021619621524474046], [0.9792789935511154, 0.02072100644888466], [0.9789089755788138, 0.021091024421186173], [0.9770060260069775, 0.02299397399302252], [0.9772703245586214, 0.02272967544137858], [0.978380378475526, 0.021619621524474046]]
}
def main(args):
  print('running module {}'.format(FLAGS.module))
  
  mLogger.enable_debugs(FLAGS.debugs.split(','))
    
  if FLAGS.module == 'ctr_dist':
    plot_clicks(FLAGS.train_clicks_pickle_path, FLAGS.topk, [4,5,6])
#     plot_two_clicks(FLAGS.train_clicks_pickle_path, FLAGS.train_clicks_pickle_path.replace('_5level', ''), FLAGS.topk, 3)
#     paths = ['/Users/aliv/MySpace/_DataSets/LTR/Microsoft/MSLR-WEB30K/clicks/clicks.train.2097154.trust_1_top50.pkl']
#     plot_multiple_separate_clicks(paths, FLAGS.topk, [30])

#     plot_clicks('/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/clicks/clicks.train.250000.trust_1_top20.pkl', 20, [1])
    return
  else:
    data = get_dataset_from_json_info(
                        FLAGS.dataset_name,
                        FLAGS.datasets_info_path,
                      ).get_data_folds()[FLAGS.data_fold]
    data.read_data()
    train_q, train_d = data_stats(data.train)
    test_q, test_d = data_stats(data.test)
    valid_q, valid_d = data_stats(data.valid)
    mLogger.mLoggers['data'].debug('data has {} queries and {} docs/query.'.format(train_q+test_q+valid_q, (train_d+test_d+valid_d)/(train_q+test_q+valid_q)))

  
  
  if FLAGS.module == 'oracle_weights':
    with open(FLAGS.ranks_pickle_path, 'rb') as f:
      ranks = pickle.load(f, encoding='latin1')['train']
    clicks_json = _get_pbm_affine_json()
    topk = clicks_json['topk'] if 'topk' in clicks_json else FLAGS.default_topk
    level2prob = eval(clicks_json['level2prob'])
    weights = get_oracle_weights(ranks, data.train, topk, level2prob)
    print(weights)
    return
    
  if FLAGS.module == 'simulate_clicks':
    with open(FLAGS.ranks_pickle_path, 'rb') as f:
#       all_ranks_pickle = pickle.load(f)
      all_ranks_pickle = pickle.load(f, encoding='latin1')
      
    simulate_clicks(data = data, 
                    click_count = eval(FLAGS.click_count), 
                    click_policy_path = FLAGS.click_policy_path, 
                    all_ranks_pickle = all_ranks_pickle, 
                    output_dir = os.path.dirname(FLAGS.ranks_pickle_path), 
                    default_topk = FLAGS.default_topk)
    return
    
  _affine_json = _get_pbm_affine_json()
  correction_kwargs_map = {
      'full_info':{'data':data, 'json':_get_pbm_affine_json()},
      'no_correction':{'alpha':np.ones([_affine_json['topk']],dtype=np.float64), 'beta':np.zeros([_affine_json['topk']],dtype=np.float64)},
      'pbm_affine':{'json':_affine_json},
      'pbm_soft_clustering':{'mixture':FLAGS.mixture, 'enhance_mult':FLAGS.enhance_data, 'enhance_dist':FLAGS.enhance_distribution, 'soft':'soft', 'n_components':FLAGS.n_components, 'n_init':1, 'warm_start':False, 'max_iter':100},
      'pbm_soft_clustering_g':{'mixture':'gaussian', 'enhance_mult':FLAGS.enhance_data, 'enhance_dist':FLAGS.enhance_distribution, 'soft':'soft', 'n_components':FLAGS.n_components, 'n_init':1, 'warm_start':False, 'max_iter':100},
      'pbm_soft_clustering_b':{'mixture':'binomial', 'enhance_mult':FLAGS.enhance_data, 'enhance_dist':FLAGS.enhance_distribution, 'soft':'soft', 'n_components':FLAGS.n_components, 'n_init':1, 'warm_start':False, 'max_iter':100},
#           'pbm_soft2_clustering':{'mixture':FLAGS.mixture, 'soft':'soft2', 'n_components':FLAGS.n_components, 'n_init':1, 'warm_start':False, 'max_iter':100},
          'pbm_hard_clustering':{'mixture':FLAGS.mixture, 'enhance_mult':FLAGS.enhance_data, 'enhance_dist':FLAGS.enhance_distribution, 'soft':'hard', 'n_components':FLAGS.n_components, 'n_init':1, 'warm_start':False, 'max_iter':100},
      'pbm_oracle_soft_clustering':{'oracle':{'weights':oracle_rel_weights[FLAGS.dataset_name], 'json':_get_pbm_affine_json()}, 'enhance':FLAGS.enhance_data, 'soft':'soft', 'n_components':2, 'n_init':1, 'warm_start':False, 'max_iter':100}
      }
      
  
  learning_kwargs_map = {
      'DNN':{'loss_function_str':FLAGS.DNN_loss_function_str, 'max_train_iterations':FLAGS.DNN_max_train_iterations, 'batch_size':FLAGS.DNN_batch_size, 'learning_rate':eval(FLAGS.DNN_learning_rate), 'embed_size':FLAGS.DNN_embed_size},
      'lambdamart':{'loss_function_str':FLAGS.DNN_loss_function_str, 'early_stopping_rounds':FLAGS.lambdamart_early_stopping_rounds, 'eval_at':eval(FLAGS.lambdamart_eval_at)}
      }
  
  regression_kwargs_map = {
      'DNN':{'loss_function_str':FLAGS.regression_loss_function_str, 'max_train_iterations':FLAGS.regression_DNN_max_train_iterations, 'batch_size':FLAGS.regression_DNN_batch_size, 'learning_rate':eval(FLAGS.regression_DNN_learning_rate), 'embed_size':FLAGS.regression_DNN_embed_size},
      'lambdamart':{'loss_function_str':FLAGS.regression_loss_function_str, 'early_stopping_rounds':FLAGS.regression_lambdamart_early_stopping_rounds, 'eval_at':eval(FLAGS.regression_lambdamart_eval_at)}
      }
      
  if FLAGS.module == 'compare':
    mcorrection = {'pbm_affine':PBMAffineCorrection,
                'pbm_soft_clustering':PBMClusteringCorrection,
                'pbm_soft_clustering_g':PBMClusteringCorrection,
                'pbm_soft_clustering_b':PBMClusteringCorrection,
#                 'pbm_soft2_clustering':PBMClusteringCorrection,
#                 'pbm_hard_clustering':PBMClusteringCorrection,
#                 'pbm_oracle_soft_clustering':PBMClusteringCorrection
                }

    corrections = {}
#     for correction_method in correction_kwargs_map:
    for correction_method in ['pbm_affine', 'pbm_soft_clustering']:
#     for correction_method in ['pbm_soft_clustering_g', 'pbm_soft_clustering_b']:
#     for correction_method in ['pbm_affine']:
      print(correction_method)
      kwargs = correction_kwargs_map[correction_method]
      print(kwargs)
      corrections[correction_method] = mcorrection[correction_method](**kwargs)
      corrections[correction_method].add_clicks_pickle_path('train', FLAGS.train_clicks_pickle_path)
#     compare_with_gold(data, corrections, 3, 60)
    plot_corrected(data, corrections, [0,2,5,8], '/Users/aliv/Dropbox/MyPapers/2020-clustering-cltr/sections/figure/8000000_dcm_1_top10')
      
  if FLAGS.module == 'train_and_test':
    train_and_test(data = data, 
                    train_clicks_pickle_path = FLAGS.train_clicks_pickle_path, 
                    correction_method = FLAGS.correction_method,
#                     correction_kwargs = {'alpha': np.ones([50], dtype=np.float64), 'beta': np.zeros([50], dtype=np.float64)}, 
#                     correction_kwargs = {'alpha': np.array([0.3203328114167142, 0.9974268197866216, 0.7019049532258751, 0.3390626177211712, 0.2284738983227512, 0.15731807333390377, 0.11613244075963333, 0.09017590991363512, 0.07146232588517047, 0.05864023900171775]), 'beta':np.array([0.6796671841809837, 0.0021485382555437786, 8.53185856098904e-07, 1.1606727898143262e-07, 3.974305120817835e-08, 1.928116384354399e-08, 1.5015814080879785e-08, 9.061990417267106e-09, 4.894499161944713e-09, 3.4360227001328637e-09])}, 
                    correction_kwargs = correction_kwargs_map[FLAGS.correction_method], 
                    learning_algorithm = FLAGS.learning_algorithm,
                    learning_kwargs = learning_kwargs_map[FLAGS.learning_algorithm.replace('EM_','')],
                    regression_fn = FLAGS.regression_function,
                    regression_kwargs = regression_kwargs_map[FLAGS.regression_function],
                    output_path = FLAGS.output_path,
                    slurm_job_id = FLAGS.slurm_job_id,
                    EM_iterations = FLAGS.EM_max_iterations,
                    binary_rel = FLAGS.binary_rel)

if __name__ == '__main__':
  app.run(main)
#   app.run(main_arr)