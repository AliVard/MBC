'''
Created on 28 Jun 2020

@author: aliv
'''
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
cmap = plt.get_cmap('jet')


_G_COLORS = ['#d55e00', '#009e73', '#cc79a7', '#56b4e9']
_G_MARKERS = ['>', 'o', 'x', '+']
_G_MARK_EVERY = 7
_G_MARKER_SIZE = 8

legend_info = {
  'affine':{
        'linestyle': '-',
        'color': _G_COLORS[0],
        'marker': _G_MARKERS[0],
        'markersize': _G_MARKER_SIZE,
        'markevery': _G_MARK_EVERY,
        'fillstyle': 'none',
        'label': 'Affine',
        'linewidth': 2.,
      },
  'ips':{
        'linestyle': '-',
        'color': _G_COLORS[2],
        'marker': _G_MARKERS[2],
        'markersize': _G_MARKER_SIZE,
        'markevery': _G_MARK_EVERY,
        'fillstyle': 'none',
        'label': 'IPS',
        'linewidth': 2.,
      },
  'cluster':{
        'linestyle': '-',
        'color': _G_COLORS[1],
        'marker': _G_MARKERS[1],
        'markersize': _G_MARKER_SIZE,
        'markevery': _G_MARK_EVERY,
        'fillstyle': 'none',
        'label': 'Clustering',
        'linewidth': 2.,
      },
  'cluster -':{
        'linestyle': '-',
        'color': _G_COLORS[3],
        'marker': _G_MARKERS[3],
        'markersize': _G_MARKER_SIZE,
        'markevery': _G_MARK_EVERY,
        'fillstyle': 'none',
        'label': 'Cluster (non-relevant)',
        'linewidth': 2.,
      },
  }
def Binom_prob(n, k, p):
  coeff = np.log(binom(n,k))
  log_prob = coeff + (k * np.log(p)) + ((n-k)*np.log(1.-p))
  return np.exp(log_prob)

class BaseMethod(object):
  def __init__(self, alpha, beta, n=20):
    self.alpha = alpha
    self.beta = beta
    self.n = n
    
  def _rel_prob(self, k):
    prob = Binom_prob(self.n, k, self.alpha + self.beta)
#     print('rel @ {} : {}'.format(k, prob))
    return prob
  
  def _nonrel_prob(self, k):
    prob = Binom_prob(self.n, k, self.beta)
#     print('nonrel @ {} : {}'.format(k, prob))
    return prob
  
  def _mean(self, prob_fn):
    pass
    
  def rel_mean(self):
    return self._mean(self._rel_prob)
  
  def nonrel_mean(self):
    return self._mean(self._nonrel_prob)


class IPS(BaseMethod):
  def __init__(self, alpha, beta, n):
    super(IPS, self).__init__(alpha, beta, n)
    
  def _mean(self, prob_fn):
    mu = 0
    for k in range(self.n + 1):
      corrected = ((k * 1. / self.n)) / (self.alpha+2*self.beta)
#       print('{} -> corrected: {}, prob:{}, product:{}'.format(k, corrected, prob_fn(k), corrected * prob_fn(k)))
      mu += corrected * prob_fn(k)
    return mu
  
  def rel_std(self):
    prob = self.alpha + self.beta
    return prob * (1. - prob) / self.n / (self.alpha+2*self.beta)**2
  def nonrel_std(self):
    prob = self.beta
    return prob * (1. - prob) / self.n / (self.alpha+2*self.beta)**2
    

class Affine(BaseMethod):
  def __init__(self, alpha, beta, n, non_negative = False):
    super(Affine, self).__init__(alpha, beta, n)
    self.non_negative = non_negative
    
  def _mean(self, prob_fn):
    mu = 0
    for k in range(self.n + 1):
      corrected = ((k * 1. / self.n) - self.beta) / self.alpha
      if self.non_negative and corrected < 0.:
        corrected = 0.
#       print('{} -> corrected: {}, prob:{}, product:{}'.format(k, corrected, prob_fn(k), corrected * prob_fn(k)))
      mu += corrected * prob_fn(k)
    return mu
  
  def rel_std(self):
    prob = self.alpha + self.beta
    return prob * (1. - prob) / self.n / self.alpha**2
  def nonrel_std(self):
    prob = self.beta
    return prob * (1. - prob) / self.n / self.alpha**2
    

class Cluster(BaseMethod):
  def __init__(self, alpha, beta, n):
    super(Cluster, self).__init__(alpha, beta, n)
    kk = np.log((1.-beta)/(1.-alpha-beta))
    nn = np.log((alpha+beta)/beta)
    self.threshold = kk / (nn + kk)
    
  def _mean(self, prob_fn):
    mu = 0
    for k in range(self.n + 1):
      corrected = 1. if k > (self.n * self.threshold) else 0.
      if corrected != 0.:
        mu += corrected * prob_fn(k)
        
#     print('th:{}, n:{}, mu:{}'.format(self.n * self.threshold, self.n, mu))
    return mu
  
  def rel_std(self):
    mu = self.rel_mean()
    return mu * (1. - mu) 
  def nonrel_std(self):
    mu = self.nonrel_mean()
    return mu * (1. - mu) 
  
def compare_methods_mean(zetap, zetan, min_n, max_n):
  alpha = zetap - zetan
  beta = zetan
  ns = np.array(list(range(min_n,max_n+1)))
  affine_p = np.zeros_like(ns, dtype=np.float64)
  affine_n = np.zeros_like(ns, dtype=np.float64)
  ips_p = np.zeros_like(ns, dtype=np.float64)
  ips_n = np.zeros_like(ns, dtype=np.float64)
  cluster_p = np.zeros_like(ns, dtype=np.float64)
  cluster_n = np.zeros_like(ns, dtype=np.float64)
  
  for i, n in enumerate(ns):
    method = Affine(alpha, beta, n , False)
    affine_p[i] = method.rel_mean()
    affine_n[i] = method.nonrel_mean()
    method = IPS(alpha,beta, n)
    ips_p[i] = method.rel_mean()
    ips_n[i] = method.nonrel_mean()
    method = Cluster(alpha, beta, n)
    cluster_p[i] = method.rel_mean()
    cluster_n[i] = method.nonrel_mean()
    
#   plt.subplot(1,2,1)
#   plt.title('$\alpha={}\quad\&\quad\beta={}$'.format(alpha,beta))
#   plt.xlabel('sessions per query')
#   plt.ylim(-0.1, 1.1)
  plt.plot(ns, ips_p, **legend_info['ips'])
  plt.plot(ns, affine_p, **legend_info['affine'])
  plt.plot(ns, cluster_p, **legend_info['cluster'])
  
  
#   plt.subplot(1,2,2)
#   plt.title('non-relevant')
#   plt.xlabel('sessions per query')
#   plt.ylim(-0.1, 1.1)
  plt.plot(ns, ips_n, **legend_info['ips'])
  plt.plot(ns, affine_n, **legend_info['affine'])
  plt.plot(ns, cluster_n, **legend_info['cluster'])
#   plt.legend()
  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  plt.legend(by_label.values(), by_label.keys())
  print('$p(c^+)=%.03f, p(c^-)=%.03f$ &' %(alpha+beta, beta))
  
  
def compare_methods_std(zetap, zetan, min_n, max_n):
  alpha = zetap - zetan
  beta = zetan
  ns = np.array(list(range(min_n,max_n+1)))
  affine_p = np.zeros_like(ns, dtype=np.float64)
  affine_n = np.zeros_like(ns, dtype=np.float64)
  ips_p = np.zeros_like(ns, dtype=np.float64)
  ips_n = np.zeros_like(ns, dtype=np.float64)
  cluster_p = np.zeros_like(ns, dtype=np.float64)
  cluster_n = np.zeros_like(ns, dtype=np.float64)
  
  for i, n in enumerate(ns):
    method = Affine(alpha, beta, n , False)
    affine_p[i] = method.rel_std()
    affine_n[i] = method.nonrel_std()
    method = IPS(alpha,beta, n)
    ips_p[i] = method.rel_std()
    ips_n[i] = method.nonrel_std()
    method = Cluster(alpha, beta, n)
    cluster_p[i] = method.rel_std()
    cluster_n[i] = method.nonrel_std()
    
#   plt.subplot(1,2,1)
#   plt.title('$\alpha={}\quad\&\quad\beta={}$'.format(alpha,beta))
#   plt.xlabel('sessions per query')
#   plt.ylim(-0.1, 1.1)
  plt.plot(ns, ips_p, **legend_info['ips'])
  plt.plot(ns, affine_p, **legend_info['affine'])
  plt.plot(ns, cluster_p, **legend_info['cluster'])
  
  
#   plt.subplot(1,2,2)
#   plt.title('non-relevant')
#   plt.xlabel('sessions per query')
#   plt.ylim(-0.1, 1.1)
  plt.plot(ns, ips_n, **legend_info['ips'])
  plt.plot(ns, affine_n, **legend_info['affine'])
  plt.plot(ns, cluster_n, **legend_info['cluster'])
#   plt.legend()
  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  plt.legend(by_label.values(), by_label.keys())
  print('$p(c^+)=%.03f, p(c^-)=%.03f$ &' %(alpha+beta, beta))
  
  
def main(zetap, zetan, min_n, max_n):
#   plt.subplot(1,2,1)
  compare_methods_mean(zetap, zetan, min_n, max_n)
#   plt.subplot(1,2,2)
#   compare_methods_std(zetap, zetan, min_n, max_n)
#   plt.suptitle('P(rel)=%.03f   ----   P(non-rel)=%.03f' %(zetap, zetan))
#   plt.show()
 
def main_error():
  theta = [1./(i+1) for i in range(20)]
  ep = [0.98-(i/100.) for i in range(20)]
  en = [0.65/(i+1.) for i in range(10)]
  for i, position in enumerate([0,2,5,9]):
#     plt.subplot(1,4,position+1)
    plt.figure(i)
    compare_methods_mean(theta[position]*ep[position], theta[position]*en[position], 5, 40)
    plt.savefig('/Users/aliv/Dropbox/MyPapers/slides-2020-clustering-cltr-soos/figure/error{}.pdf'.format(i),bbox_inches='tight', pad_inches=0)

def main_std():
  theta = [1./(i+1) for i in range(20)]
  ep = [0.98-(i/100.) for i in range(20)]
  en = [0.65/(i+1.) for i in range(10)]
  for i, position in enumerate([0,2,5,9]):
#     plt.subplot(1,4,position+1)
    plt.figure(i)
    plt.ylim(-0.05,1.05)
    compare_methods_std(theta[position]*ep[position], theta[position]*en[position], 5, 40)
    plt.savefig('/Users/aliv/Dropbox/MyPapers/slides-2020-clustering-cltr-soos/figure/std{}.pdf'.format(i),bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
  main_std()
#   main_error()
#   theta = [1./(i+1) for i in range(20)]
#   ep = [0.98-(i/100.) for i in range(20)]
#   en = [0.35/(i+1.) for i in range(10)]
#   position = 4
#   main(theta[position]*ep[position], theta[position]*en[position], 5, 50)