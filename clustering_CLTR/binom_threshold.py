'''
Created on 20 Oct 2020

@author: aliv
'''
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

def prob_eq(p, n, k):
  return binom(n, k) * (p**k) * ((1.-p)**(n-k))

def prob_less(p, n, tau):
  prob = 0
  for k in range(tau):
    prob += prob_eq(p, n, k)
  return prob

def prob_greater(p, n, tau):
  prob = 0
  for k in range(tau,n+1):
    prob += prob_eq(p, n, k)
  return prob

def max_error(p0, p1, n, tau):
  prob0 = prob_greater(p0, n, tau)
  prob1 = prob_less(p1, n, tau)
  return max(prob0, prob1)

def max_error_vec(p0, p1, n):
  taus = np.array([i for i in range(n+1)])
  errs = np.zeros([n+1], dtype=np.float)
  for i, tau in enumerate(taus):
    errs[i] = max_error(p0, p1, n, tau)
    
  return taus, errs

zp = np.array([1./(i+1.) for i in range(10)]) * np.array([0.98-(i/100.) for i in range(10)])
zn = np.array([1./(i+1.) for i in range(10)]) * np.array([0.65/(i+1.) for i in range(10)])
for pos in range(10):
  p0 = zn[pos]
  p1 = zp[pos]
  n = 400
  x, y = max_error_vec(p0, p1, n)
  arg = np.argmin(y)
  print('n:{}, p0:{}, p1:{}, best tau:{}, min error:{}, <{}=1/n taus:{}'.format(n, p0, p1, x[arg], y[arg], 1./n, len(x[y<1./n])))
# plt.plot(x,y)
# plt.show()