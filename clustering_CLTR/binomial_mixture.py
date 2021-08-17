'''
Created on 28 May 2020

@author: aliv
'''
"""Binomial Mixture Model."""

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import numpy as np

from scipy import linalg

from sklearn.mixture._base import BaseMixture, _check_shape, check_random_state, _check_X
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import row_norms
from sklearn import cluster
from scipy.special import binom

from .mLogger import mLoggers

###############################################################################
# Binomial mixture shape checkers used by the BinomialMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means


def _estimate_binomial_parameters(X, resp):
    """Estimate the Binomial distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        The input data array.
        First column: number of success (k)
        Second column: number of tries (n)

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.


    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like, shape (n_components, 2)
        The centers of the current components.

    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    
#     hard_resp = np.zeros_like(resp)
#     hard_resp[np.arange(resp.shape[0]),resp.argmax(axis=1)] = 1
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
#     print(np.concatenate([X,hard_resp],1)[:20,:])
#     print(nk)
#     print(means)
    return nk, means

def _log_binomial_coeff(X):
  n_samples = X.shape[0]
  coeff = np.empty(n_samples)
  for i in range(n_samples):
    coeff[i] = np.log(binom(X[i,1],X[i,0]))
  return coeff

def _estimate_log_binomial_prob(X, means):
    """Estimate the log Binomial probability.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)

    means : array-like, shape (n_components, 2)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    
    log_prob = np.empty((n_samples, n_components))
    
    p = (means[:,0] / means[:,1])
#     p[p == 0] = np.finfo(np.float64).eps
#     p[p == 1] = 1 - np.finfo(np.float64).eps
#     q = np.log(1. - p)
#     p = np.log(p) #todo: log(0.)
    q = np.log(1. - p + np.finfo(np.float64).eps)
    p = np.log(np.finfo(np.float64).eps + p) #todo: log(0.)
    
    mild_X = X 
    coeff = _log_binomial_coeff(mild_X)
    mLoggers['enhance'].debug('coeff:{}, p:{}, q:{}'.format([(coeff.min(), mild_X[coeff.argmin(),:]), (coeff.max(), mild_X[coeff.argmax(),:])], p, q))
    for c in range(n_components):
      log_prob[:, c] =  coeff + mild_X[:,0] * p[c] + (mild_X[:,1] - mild_X[:,0]) * q[c]
      
    mLoggers['enhance'].debug('min:{}, max:{}'.format(np.min(log_prob,0), np.max(log_prob,0)))
    return log_prob



class BinomialMixture(BaseMixture):
    """Binomial Mixture.

    Representation of a Binomial mixture model probability distribution.
    This class allows to estimate the parameters of a Binomial mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    means_init : array-like, shape (n_components, 2), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.


    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, 2)
        The mean of each mixture component.

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.
    """

    def __init__(self, n_components=1, tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, 
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super(BinomialMixture, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.weights_init = weights_init
        self.means_init = means_init

    def _check_parameters(self, X):
        """Check the Binomial mixture parameters are well defined."""
        _, n_features = X.shape
        
        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)


    def _initialize(self, X, resp):
        """Initialization of the Binomial mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
        First column: number of success (k)
        Second column: number of tries (n)

        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape
        
        p = X[:,0:1] / X[:,1:2]
        resp = np.zeros((n_samples, self.n_components))
        random_state = check_random_state(self.random_state)
        label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                               random_state=random_state).fit(p).labels_
        resp[np.arange(n_samples), label] = 1

        weights, means = _estimate_binomial_parameters(X, resp)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init


    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
        First column: number of success (k)
        Second column: number of tries (n)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_ = _estimate_binomial_parameters(X, np.exp(log_resp))
        self.weights_ /= n_samples
        

    def _estimate_log_prob(self, X):
        return _estimate_log_binomial_prob(
            X, self.means_)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_'])

    def _get_parameters(self):
        return (self.weights_, self.means_)
      
    def _get_printable_parameters(self):
        return (self.weights_, self.means_[:,0]/self.means_[:,1])

    def _set_parameters(self, params):
        (self.weights_, self.means_) = params



    def _n_parameters(self):
        return int(2 * self.n_components - 1)

    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_prob(X) + self._estimate_log_weights()


    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.means_.shape[1])
        return self._estimate_weighted_log_prob(X).argmax(axis=1)
#         return self._estimate_log_prob(X).argmax(axis=1)
      
    def soft_predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        self._check_is_fitted()
#         prob = np.exp(self._estimate_log_prob(X))
        prob = np.exp(self._estimate_weighted_log_prob(X))
        prob_sum = np.sum(prob,1)
        return prob / prob_sum[:,None]

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()



class OracleBinomialMixture(BinomialMixture):
  
    def __init__(self, means, n_components=1, tol=1e-3,
                 reg_covar=1e-6, max_iter=100, 
                 weights_init=None, means_init=None, n_init=1, init_params='kmeans',
                 random_state=None, warm_start=False, 
                 verbose=0, verbose_interval=10):
        super(OracleBinomialMixture, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.weights_init = weights_init
        self.means_init = means_init
        self.means = np.concatenate([means[:,np.newaxis],np.ones_like(means[:,np.newaxis])],1)

    def _initialize(self, X, resp):
        """Initialization of the Binomial mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
        First column: number of success (k)
        Second column: number of tries (n)

        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape
        
        probs = _estimate_log_binomial_prob(X, self.means)
        label = probs.argmax(axis=1)
        resp[np.arange(n_samples), label] = 1

        weights, _ = _estimate_binomial_parameters(X, resp)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = self.means
        
        
    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
        First column: number of success (k)
        Second column: number of tries (n)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, _ = _estimate_binomial_parameters(X, np.exp(log_resp))
        self.weights_ /= n_samples
        