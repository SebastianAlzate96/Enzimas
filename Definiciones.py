# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 18:40:57 2023

@author: sebas
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.stats import norm
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import cluster
from hmmlearn import hmm
from hmmlearn.base import _BaseHMM

"""
----------------------------------------------------------------------------
             # Definicion para la funcion Poisson HMM #
----------------------------------------------------------------------------  
"""

def _check_and_set_poisson_n_features(model, X):
    _, n_features = X.shape
    if hasattr(model, "n_features") and model.n_features != n_features:
        raise ValueError("Unexpected number of dimensions, got {} but "
                         "expected {}".format(n_features, model.n_features))
    model.n_features = n_features



class PoissonHMM(_BaseHMM):
   
    def __init__(self, n_components=1,
                 #min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 #covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stm", init_params="stm"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        #self.covariance_type = covariance_type
        #self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        #self.covars_prior = covars_prior
        #self.covars_weight = covars_weight

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
           
        }

    def _init(self, X, lengths=None):
        _check_and_set_poisson_n_features(self, X)
        super()._init(X,lengths=lengths)

        if self._needs_init("m", "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
 #       if self._needs_init("c", "covars_"):
  #          cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
   #         if not cv.shape:
    #            cv.shape = (1, 1)
     #       self.covars_ = \
      #          _utils.distribute_covar_matrix_to_match_covariance_type(
       #             cv, self.covariance_type, self.n_components).copy()



    def _check(self):
        super()._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]
       

    def _compute_log_likelihood(self, X):
        return sp.stats.poisson.logpmf(X[:, None, :],self.means_).sum(axis=-1)
   

    def _generate_sample_from_state(self, state, random_state=None):
        random_state = check_random_state(random_state)
        return random_state.poisson(
            self.means_[state]
        )
     
       
       
    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        #if self.covariance_type in ('tied', 'full'):
         #   stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
          #                                 self.n_features))
        return stats
   

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params: #or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)



    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, None]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))


"""
----------------------------------------------------------------------------
             # Definicion para las graficas de los residuales #
----------------------------------------------------------------------------  
"""

def pois_HMM_lforward(x,gamma,delta,lamb,m):
  n=len(x)
  delta=np.array(delta)
  lamb=np.array(lamb)
  lalpha=np.empty([m,n])
  foo=delta*poisson.pmf(x[0],lamb)
  sumfoo=sum(foo)
  lscale=np.log(sumfoo)
  foo=foo/sumfoo
  lalpha[:,0]=lscale+np.log(foo)
  for i in range(1,n):
        foo=np.dot(foo,gamma*poisson.pmf(x[i],lamb))
        sumfoo=sum(foo)
        lscale=np.log(sumfoo)
        foo=foo/sumfoo
        lalpha[:,i]=lscale+np.log(foo)
  return(lalpha)


def pois_HMM_lbackward(x,gamma,lamb,m):
  n=len(x)
  lamb=np.array(lamb)
  lbeta=np.empty([m,n])
  lbeta[:,n-1]=np.zeros(m)
  foo=np.ones(m) * 1/(m)
  lscale=np.log(m)
  for i in np.arange(n-2,-1,-1):
        foo=np.dot(gamma,poisson.pmf(x[i+1],lamb)*foo)
        lbeta[:,i]=np.log(foo)+lscale
        sumfoo_=sum(foo)
        foo=foo/sumfoo_
        lscale=lscale+np.log(sumfoo_)
  return(lbeta)


def pois_HMM_conditional(xc,x,delta,gamma,lamb,m):
    n=len(x)
    x=x.astype(int)
    lamb=np.array(lamb)
    nxc=len(xc)
    dxc=np.empty([nxc,n])
    Px=np.empty([m,nxc])
    for j in range(nxc):
        Px[:,j]=poisson.pmf(xc[j],lamb)
    la=pois_HMM_lforward(x,gamma,delta,lamb,m)
    lb=pois_HMM_lbackward(x,gamma,lamb,m)
    b=np.empty([m,1])
    b[:,0]=np.log(delta)
    la=np.hstack((b,la))
    lafact=np.amax(la,0)
    lbfact=np.amax(lb,0)
    for i in range(n):
        foo=np.dot(np.exp(la[:,i]-lafact[i]),gamma)*np.exp(lb[:,i]-lbfact[i])
        foo=foo/sum(foo)
        dxc[:,i]=np.dot(foo,Px)
    return(dxc)

def pois_HMM_pseudo_residuals(x,delta,gamma,lamb,m):
  n=len(x)
  x=x.astype(int)
  cdists=pois_HMM_conditional(range(max(x)+1),x,delta,gamma,lamb,m)
  cumdists=np.vstack((np.zeros(n),np.cumsum(cdists,0),np.array([np.nan]*n)))
  ulo=np.array([np.nan]*n)
  uhi=np.array([np.nan]*n)
  for i in range(n):
    ulo[i]=cumdists[x[i],i]
    uhi[i]=cumdists[x[i]+1,i]
  umi=0.5*(ulo+uhi)
  nspr=norm.ppf(np.vstack((ulo,umi,uhi)))
  return(nspr[1,:],umi)


"""
-------------------------------------------------------------------------------
                        #  Duracion de rafagas #
-------------------------------------------------------------------------------  
"""


def duracion_rafaga(markovmodel_P, Y):
    markovmodel_P.fit(Y) #Estima los parámetros del modelo
    means_ = markovmodel_P.means_
    means_sort = np.sort(means_[:,0])
    z= list(means_[:,0]).index(means_sort[0])
    
    states=markovmodel_P.predict(Y) #Encuentre la secuencia de estado más probable correspondiente a X.
    states = np.append(states, z)
    len(states)
    
    clus=[]
    cl=[0]
    
    for n in range(len(states)-1):
        if states[n]!=z:
            cl[0] +=1
            if states[n+1]==z:
                clus=np.append(clus,cl)
        else:
            cl=[0]
    
    print(f'El promedio de rafagas es: {np.round(np.mean(clus),2)}')
    #print(clus[:500])
    print(f'El minimo es: {min(clus)} y el maximo es: {max(clus)}')