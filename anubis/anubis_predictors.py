#!/usr/bin/env python3

#############################################################
#
# anubis_predictors.py
#
# Author : Miravet-Verde, Samuel
#
#############################################################

#####################
#   PACKAGES LOAD   #
#####################

import sys
import copy
import numpy as np
import pandas as pd
import seaborn as sns

from pylab import cm
from math import log1p, log, pi, sqrt
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex

from scipy import stats # includes poisson and gamma

from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture as BGMM


def method_iterator(datasets, me):

    pass



# Poisson probabilities for no insertion and saturation
def probability_no_insertions(data, r, method='poisson'):
    if method=='poisson':
        Ls = np.array(data.metric['L'])
        return stats.poisson.pmf(0, r*Ls)

def probability_saturation(data, r, method='poisson'):
    if method=='poisson':
        Ls = list(data.metric['L'])
        x = []
        for l in Ls:
            x.append(stats.poisson.pmf(int(r*l), r*l))
        return x


class predictor(object):
    def __init__(self, data=None,
                 n_components=3, covariance_type='full', max_components=10, _expected_labels={3:['E', 'F', 'N']},
                 autoparam=False,
                 metric='dens'):

        self.data = copy.copy(data)

        self.n_components = n_components
        self.covariance_type = covariance_type
        self._explabels = _expected_labels

        self.autoparam = autoparam
        self.max_components = max_components

        self.metric = metric
        self.model = None

        # Define X (input data)
        if data is not None:
            if data.metric is not False:
                self.X = self.data.metric_values[self.metric].reshape(-1,1)
                self.ides = self.data.metric_values['ides']

    def _return_r(self, prior, metric, **kwargs):
        if type(prior)==list:
            # Assume annotation in annotation dictionary
            vals = self.data.metric[metric].loc[prior]
            print(111)
            return np.mean(vals)
        elif type(prior)==dict:
            vals = self.data._single_metric_by(metric=metric, by='annotation',
                                               annotation_dict=prior,
                                               annotation_shortening=False)
            return np.mean(list(vals.values()))
        elif type(prior)==float or type(prior)==int:
            return float(prior)

    def define_priors(self, prior, **kwargs):
        params = {}
        __goldset = False
        for k,v in prior.items():
            if v=='0' or v==0:
                _r = self._return_r(prior=0, metric=kwargs.get('metric', self.metric))  # if value is number
            elif v=='intergenic':
                _r = self._return_r(prior=self.data.intergenic_annotation(**kwargs), metric=kwargs.get('metric', self.metric))   # intergenic
            elif v=='goldset':
                # by goldset
                if k=='E' and len(self.data.goldE)>0:
                    _r = self._return_r(prior=self.data.goldE, metric=kwargs.get('metric', self.metric))
                elif (k=='N' or k=='NE') and len(self.data.goldNE)>0:
                    _r = self._return_r(prior=self.data.goldNE, metric=kwargs.get('metric', self.metric))
                else:
                    sys.exit('No prior distribution for r given. Define prior dictionary or goldset to the dataset')
                __goldset = True
            elif type(v)==list:
                _r = self._return_r(prior=v, metric=kwargs.get('metric', self.metric))   # if a list of genes is passed
            elif type(v)==float or type(v)==int:
                _r = self._return_r(prior=v, metric=kwargs.get('metric', self.metric))
            else:
                print("Not recognized prior format for {}".format(k))
            params[k] = _r
        params = {k: v for k, v in sorted(params.items(), key=lambda item: item[1])}
        return params, __goldset

    def probabilities(self, r, model, metrics=['I', 'L'], mode='mass', _general=None, inverse=False):
        """ <mode> can be 'mass' for mass function and 'acc' for accumulated probability """
        models = {'poisson':stats.poisson, 'lognorm':stats.lognorm, 'gamma':stats.gamma,
                  'gumbel_r':stats.gumbel_r, 'gumbel_l':stats.gumbel_l}
        if model not in models:
            sys.exit('{} model not recognized as implemented model, please provide one from {}'.format(model, list(models.keys())))
        else:
            if _general is not None:
                ks = _general
                mu = np.array([100*r]*len(ks))
            else:
                mu = self.data.metric[metrics[1]]*r
                ks = self.data.metric[metrics[0]]
            if mode=='acc':
                if inverse:
                    prob = models[model].sf(ks, mu) # sf = Survival function == 1 - cdf, sf is sometimes more accurate.
                else:
                    prob = models[model].cdf(ks, mu)
            else:
                if model=='poisson':
                    prob = models[model].pmf(ks, mu)
                else:
                    prob = models[model].pdf(ks, mu)
                if inverse:
                    return 1-prob
            return prob

    def _assign_class(self, results, criterion=0, thr=0.01, labels=['E', 'N'], params=None, foldchange=2, na='F', header='class'):
        """
        Assign class based on a dataframe of results returned by a predictor.
        It will use as reference the labels included and it will add the label <na> if not computed.

        <criterion> can be any of the following ones, each designed for different assumptions. If this
        argument is in list format it will include all the different classes as class_<criterion>:
            0 - 'absolute' : used in lluch senar 2015
            1 - 'threshold' : from Osterman and Gredes book
            2 - 'foldchange' : from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2792183/pdf/2308.pdf (Langridge et al 2009, essentiality in Salmonella typhi)

        <params> include the r for Poisson as {'E':0.01, 'N':0.8}
        """
        crit = {0:'absolute', 1:'threshold', 2:'foldchange'}
        tirc = {v:k for k, v in crit.items()}

        if type(criterion)!=list:
            if criterion not in crit and criterion not in tirc:
                sys.exit('Not valid criterion')
            else:
                if criterion in crit:
                    criterion=crit[criterion]

            # Work with different criteria
            predclass = []
            if criterion=='absolute':
                for m0, p0, p1 in zip(results[self.metric], results['P({})'.format(labels[0])], results['P({})'.format(labels[1])]):
                    if m0<params[labels[0]]:
                        predclass.append(labels[0])
                    elif params[labels[-1]]<m0:
                        predclass.append(labels[-1])
                    else:
                        if p0>0 and p1==0:
                            predclass.append(labels[0])
                        elif p0==0 and p1>0:
                            predclass.append(labels[-1])
                        elif (p0>0 and p1>0) or (p0==0 and p1==0):
                            predclass.append(na)
                        else:
                            print(m0, p0, p1)
                            predclass.append(na)
            elif criterion=='threshold':
                for m0, p0, p1 in zip(results[self.metric], results['P({})'.format(labels[0])], results['P({})'.format(labels[1])]):
                    if p0>=thr:
                        predclass.append(labels[0])
                    elif p1>=thr:
                        predclass.append(labels[-1])
                    else:
                        predclass.append(na)
            elif criterion=='foldchange':
                # Approach used with gamma distributions from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2792183/pdf/2308.pdf
                if 'LLR [log2(E/N)]' in results.columns:
                    for m0, p0 in zip(results[self.metric], results['LLR [log2(E/N)]']):
                        if p0>=foldchange:
                            predclass.append('E')
                        elif p0<=-foldchange:
                            predclass.append('N')
                        else:
                            predclass.append(na)
                else:
                    p0dist = results['P({})'.format(labels[0])]
                    p1dist = results['P({})'.format(labels[1])]
                    pseudocount0 = min(p0dist[p0dist>0])
                    pseudocount1 = min(p1dist[p1dist>0])
                    psc = min([pseudocount0, pseudocount1])
                    for m0, p0, p1 in zip(results[self.metric], results['P({})'.format(labels[0])], results['P({})'.format(labels[1])]):
                        log2foldchange = np.log2((p0+psc)/(p1+psc))
                        if log2foldchange>=foldchange:
                            predclass.append(labels[0])
                        elif log2foldchange<=-foldchange:
                            predclass.append(labels[-1])
                        else:
                            predclass.append(na)
            else:
                sys.exit('Not compatible criterion')
            results[header] = predclass
        else:
            for c in criterion:
                results = self._assign_class(results, c, thr=thr, labels=labels, header='{}_{}'.format(header, c))
        return results


class __statsmodel__(predictor):
    """
    Class to perform mixture models from stats classes
    """
    def __predict__(self, models=['lognorm', 'poisson'],
                    prior={'E':'goldset', 'N':'goldset'},
                    save=False, save_plot=False, show_plot=True,
                    thr = 0.01, criteria='absolute', foldchange=2,
                    probabilities=0,
                    **kwargs):
        """
        You can set different models for different priors, default:
        lognorm for E genes from goldset (defined in priors) and
        poisson for NE genes from goldset
        """
        if type(models)==str:
            models = [models]*len(prior)

        # define priors
        params, __goldset = self.define_priors(prior, **kwargs)

        # Predict probabilities for the labels
        pnis = probability_no_insertions(self.data, max(params.values()))
        psat = probability_saturation(self.data, max(params.values()))
        header = ['identifier', 'left ann', 'right ann', 'strand', 'len', 'len_considered', 'goldset', self.metric, 'P(no insertions)', 'P(saturation)']
        rs = {}
        c = 0
        for ide, met, pni, psa, used_len in zip(self.ides, self.X, pnis, psat, self.data.metric['L']):
            st, en, strand = self.data.annotation.get(ide, ['NA', 'NA', 'NA'])
            try:
                l = abs(en-st+1)
            except:
                l = 'NA'
            if __goldset:
                if ide in self.data.goldE:
                    gs = 'E'
                elif ide in self.data.goldNE:
                    gs = 'N'
                else:
                    gs = 'u'
            else:
                gs = 'u'
            rs[c] = [ide, st, en, strand, l, used_len, gs, met[0], round(pni,3), round(psa,3)]
            c+=1
        rs = pd.DataFrame.from_dict(rs, orient='index')
        rs.columns = header

        # Predict probabilities for the labels
        if probabilities==0:
            _probmode = 'acc'
        else:
            _probmode = 'mass'
        for class_ide, r in params.items():
            if class_ide=='E' and _probmode=='acc':
                rs['P({})'.format(class_ide)] = self.probabilities(r=r, model=models[0],
                                                                   metrics=kwargs.get('metrics', ['I', 'L']),
                                                                   mode=kwargs.get('mode', _probmode), inverse=True)
            else:
                rs['P({})'.format(class_ide)] = self.probabilities(r=r, model=models[-1],
                                                                   metrics=kwargs.get('metrics', ['I', 'L']),
                                                                   mode=kwargs.get('mode', _probmode), inverse=False)

        if len(prior)==2 and 'E' in prior and 'N' in prior:
            pc = rs[['P(E)', 'P(N)']].values
            pc = pc[pc>0].min()
            rs['LLR [log2(E/N)]'] = np.log2((rs['P(E)']+pc)/(rs['P(N)']+pc))

        # assign class
        rs = self._assign_class(results=rs, criterion=criteria, thr=thr, labels=sorted(list(prior.keys())), params=params, foldchange=foldchange, na='F', header='class')

        # Plot report
        if show_plot or save_plot:
            if 'dens' in self.metric:
                lim = [0, 1]
                x = np.array(range(0, 100, 1))
            else:
                lim = [0, max(self.X)+np.std(self.X)]
                x = np.array(range(0, int(max(self.X)*100), 1))
            my_cmap = sns.color_palette('mako', len(params)).as_hex()
            fig = plt.figure(figsize=(6, 8))

            ax = fig.add_subplot(211)
            ax.hist(self.X, 30, density=True, histtype='stepfilled', alpha=0.4)
            ax.set_xlabel('{}'.format(self.metric))
            ax.set_ylabel('$Frequency$')
            ax.set_xlim(lim)

            # fit with curve_fit
            ax2 = fig.add_subplot(212)
            _col_iter = 0
            for class_ide, r in params.items():
                if class_ide=='E' and _probmode=='acc':
                    probs = self.probabilities(r=r, model=models[0],
                                               metrics=kwargs.get('metrics', ['I', 'L']),
                                               mode=kwargs.get('mode', _probmode), _general=x, inverse=True)
                else:
                    probs = self.probabilities(r=r, model=models[-1],
                                               metrics=kwargs.get('metrics', ['I', 'L']),
                                               mode=kwargs.get('mode', _probmode), _general=x, inverse=False)
                # ax2.plot(x/100, stats.poisson.pmf(x, r*100), color=my_cmap[_col_iter], linestyle='--', linewidth=5, label='P({})'.format(class_ide))
                ax2.plot(x/100, probs, color=my_cmap[_col_iter], linestyle='--', linewidth=5, label='P({})'.format(class_ide))
                ax.axvline(r, c=my_cmap[_col_iter], label='r{}'.format(class_ide))
                ax2.axvline(r, c=my_cmap[_col_iter], label='r{}'.format(class_ide))
                _col_iter+=1
            ax2.set_xlabel('{}'.format(self.metric))
            ax2.set_ylabel('$p(x)$')
            ax2.set_xlim(lim)
            ax2.legend()

            if save_plot:
                plt.savefig(save_plot)
            if show_plot:
                plt.show()
            else:
                plt.close()

        # Results
        self.results = rs
        if save:
            self.results.to_excel(save)
        return self.results


class Poisson(__statsmodel__):
    """
    Poisson essentiality estimation as presented in Lluch-Senar 2015.
    We also include different assumptions not based on gold set definition
    """

    def predict(self, prior={'E':'goldset', 'N':'goldset'},
                save=False, save_plot=False, show_plot=True,
                thr = 0.01, criteria='absolute', foldchange=2,
                probabilities=0, **kwargs):
        """
        r = dictionary of where to extract the parameter and the label
            {class:annot} --> annot can be 'goldset', 'intergenic' or '0' or a list of annotations present in the object dataset
            examples:
                {'E':0, 'N':'intergenic'} --> assumes metric=0 for E genes and metric from intergenic annotations
                {'E':0, 'J':0.1} --> E centered in 0, J centered in 0.1

        criteria = based on threshold

        probabilities = 0 --> use 1-cdf for E; cdf for NE
        probabilities = 1 --> use pmf for E and NE
        """
        return self.__predict__(models='poisson', prior=prior,
                                save=save, save_plot=save_plot, show_plot=show_plot,
                                thr = thr, criteria=criteria, foldchange=foldchange,
                                probabilities=probabilities, **kwargs)


class Gamma(__statsmodel__):
    """
    Gamma essentiality estimation as presented in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5452409/
    Defining the ABC of gene essentiality in streptococci (Amelia R. L. Charbonneau, 2017)
    """

    def predict(self, prior={'E':'goldset', 'N':'goldset'},
                save=False, save_plot=False, show_plot=True,
                thr = 0.01, criteria='foldchange', foldchange=2,
                probabilities=0, **kwargs):
        """
        r = dictionary of where to extract the parameter and the label
            {class:annot} --> annot can be 'goldset', 'intergenic' or '0' or a list of annotations present in the object dataset
            examples:
                {'E':0, 'N':'intergenic'} --> assumes metric=0 for E genes and metric from intergenic annotations
                {'E':0, 'J':0.1} --> E centered in 0, J centered in 0.1

        criteria = based on threshold

        probabilities = 0 --> use 1-cdf for E; cdf for NE
        probabilities = 1 --> use pmf for E and NE
        """
        return self.__predict__(models='gamma', prior=prior,
                                save=save, save_plot=save_plot, show_plot=show_plot,
                                thr = thr, criteria=criteria, foldchange=foldchange,
                                probabilities=probabilities, **kwargs)

class Gumbel(__statsmodel__):
    """
    Adjust for a right-skewed Gumbel for E genes and left-skewed Gumbel for NE genes
    From DeJesus, M.A., Zhang, Y.J., Sassettti, C.M., Rubin, E.J., Sacchettini, J.C., and Ioerger, T.R. (2013).
    Bayesian analysis of gene essentiality based on sequencing of transposon insertion libraries. Bioinformatics, 29(6):695-703.
    """

    def predict(self, models=['gumbel_r', 'gumbel_l'], prior={'E':'goldset', 'N':'goldset'},
                save=False, save_plot=False, show_plot=True,
                thr = 0.01, criteria='foldchange', foldchange=2,
                probabilities=0, **kwargs):
        """
        Allows to combine gamma, poisson and lognorm models for a same classification
        r = dictionary of where to extract the parameter and the label
            {class:annot} --> annot can be 'goldset', 'intergenic' or '0' or a list of annotations present in the object dataset
            examples:
                {'E':0, 'N':'intergenic'} --> assumes metric=0 for E genes and metric from intergenic annotations
                {'E':0, 'J':0.1} --> E centered in 0, J centered in 0.1

        criteria = based on threshold

        probabilities = 0 --> use 1-cdf for E; cdf for NE
        probabilities = 1 --> use pmf for E and NE
        """
        return self.__predict__(models=models, prior=prior,
                                save=save, save_plot=save_plot, show_plot=show_plot,
                                thr = thr, criteria=criteria, foldchange=foldchange,
                                probabilities=probabilities, **kwargs)

class Mixture(__statsmodel__):
    """
    Allows to combine gamma, poisson and lognorm models for a same classification
    """

    def predict(self, models=['lognorm', 'poisson'], prior={'E':'goldset', 'N':'goldset'},
                save=False, save_plot=False, show_plot=True,
                thr = 0.01, criteria='foldchange', foldchange=2,
                probabilities=0, **kwargs):
        """
        Allows to combine gamma, poisson and lognorm models for a same classification
        r = dictionary of where to extract the parameter and the label
            {class:annot} --> annot can be 'goldset', 'intergenic' or '0' or a list of annotations present in the object dataset
            examples:
                {'E':0, 'N':'intergenic'} --> assumes metric=0 for E genes and metric from intergenic annotations
                {'E':0, 'J':0.1} --> E centered in 0, J centered in 0.1

        criteria = based on threshold

        probabilities = 0 --> use 1-cdf for E; cdf for NE
        probabilities = 1 --> use pmf for E and NE
        """
        return self.__predict__(models=models, prior=prior,
                                save=save, save_plot=save_plot, show_plot=show_plot,
                                thr = thr, criteria=criteria, foldchange=foldchange,
                                probabilities=probabilities, **kwargs)


class GaussianMixture(predictor):
    """
    One-dimensional Gaussian mixture model. Defined based on following examples:
    https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f
    https://www.astroml.org/book_figures_1ed/chapter4/fig_GMM_1D.html
    """
    def initialize(self, **kwargs):
        """ Accepts all arguments from https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html """
        return GMM(n_components=kwargs.get('n_components', self.n_components),
                   covariance_type=kwargs.get('covariance_type', self.covariance_type),
                   init_params=kwargs.get('init_params', 'kmeans'), max_iter=kwargs.get('max_iter', 100),
                   means_init=kwargs.get('means_init', None), n_init=kwargs.get('n_init', 1),
                   precisions_init=kwargs.get('precisions_init', None), random_state=kwargs.get('random_state', 42),
                   reg_covar=kwargs.get('reg_covar', 1e-06), tol=kwargs.get('tol', 0.001),
                   verbose=kwargs.get('verbose', 0), verbose_interval=kwargs.get('verbose_interval', 10),
                   warm_start=kwargs.get('warm_start', False), weights_init=kwargs.get('weights_init', None))

    def auto_parametrize(self, criterion='AIC', **kwargs):
        """
        Find the best fitting model without prior information
        Criteria allows to select either the model with less <AIC> (Akaike information criterion (AIC), DEFAULT)
        or with less <AIC> (Bayesian information criterion (BIC)
        """
        N = np.arange(1, kwargs.get('max_components', self.max_components))
        models = []
        for i in N:
            models.append(self.initialize(n_components=i, **kwargs).fit(self.X))

        AIC = [m.aic(self.X) for m in models]
        BIC = [m.bic(self.X) for m in models]

        if criterion=='AIC':
            M_best = models[np.argmin(AIC)]
        else:
            M_best = models[np.argmin(BIC)]
        self.model = M_best
        return M_best, AIC, BIC, models

    def fit_model(self, **kwargs):
        """ If autoparam, this will look for the best model. Run default otherwise """ 
        if self.autoparam or kwargs.get('autoparam', False):
            self.model, AIC, BIC, models = self.auto_parametrize(kwargs.get('criterion', 'AIC'), **kwargs)
            return self.model, AIC, BIC, models
        else:
            self.model = self.initialize(**kwargs).fit(self.X)
            return self.model

    def _label_sort_categories(self):
        rs = {}
        means = [i[0] for i in self.model.means_]
        n = len(means)
        if n in self._explabels and n==len(self._explabels[n]):
            class_correspondance = sorted([[m, self.model.predict([[m]])[0]] for m in means])
            ccc = set([v[1] for v in class_correspondance])  # Class correspondance checker required to avoid missing classes if very convoluted distribution
            if len(ccc)!=n:
                class_correspondance[0][1] = 1
                class_correspondance[1][1] = 2
                class_correspondance[2][1] = 0
            order = self._explabels[n]
        else:
            class_correspondance = [[round(x, 3), y] for x, y in sorted(zip(means, range(0,n)))]
            order =  ['comp{}'.format(c) for c in range(1, n+1)]
        class_correspondance = {c:m[1] for m, c in zip(class_correspondance, order)}
        return order, class_correspondance

    def report(self, **kwargs):
        """ Return dataframe of probabilities (results) """
        pm = {}
        rs = {}

        if 1==1:
            logprob = self.model.score_samples(self.X)
            responsibilities = self.model.predict_proba(self.X)
            pdf = np.exp(logprob)
            self.pdf = responsibilities * pdf[:, np.newaxis]
        else:
            self.pdf = self.model.predict_proba(self.X)
        self.labels = self.model.predict(self.X)

        c = 0

        # Prepare header
        header = ['identifier', 'left ann', 'right ann', 'strand', 'len', 'len_considered', self.metric, 'P(no insertions)', 'P(saturation)']
        order, corresp = self._label_sort_categories()
        pserroc = {v:k for k, v in corresp.items()} # reverse of corresp
        header += ['P({})'.format(i) for i in order]+['class']

        # Prob no insertions and saturation
        max_dens = max(self.model.means_)[0]
        pnis = probability_no_insertions(self.data, max_dens)
        psat = probability_saturation(self.data, max_dens)

        # populate array
        for ide, met, probs, label, pni, psa, used_len in zip(self.ides, self.X, self.pdf, self.labels, pnis, psat, self.data.metric['L']):
            st, en, strand = self.data.annotation.get(ide, ['NA', 'NA', 'NA'])
            try:
                l = abs(en-st+1)
            except:
                l = 'NA'
            prob_vector_sorted = [round(probs[corresp[e]],3) for e in order]
            rs[c] = [ide, st, en, strand, l, used_len, met[0], round(pni,3), round(psa,3)]+prob_vector_sorted+[pserroc[label]]
            c+=1

        # Return df
        rs = pd.DataFrame.from_dict(rs, orient='index')
        rs.columns = header
        return rs

    def value_predict(self, value, **kwargs):
        """ Return dataframe of probabilities (results) """
        X = [[value]]
        if not self.model:
            self.model = self.fit_model(**kwargs)
        logprob = self.model.score_samples(X)
        responsibilities = self.model.predict_proba(X)
        pdf = np.exp(logprob)
        pdf = responsibilities * pdf[:, np.newaxis]
        pdf = list(pdf[0]) # require for single pred
        label = self.model.predict(X)
        order, corresp = self._label_sort_categories()
        pserroc = {v:k for k, v in corresp.items()} # reverse of corresp

        prob_vector_sorted = {'P({})'.format(e):round(pdf[corresp[e]],3) for e in order}
        prob_vector_sorted['label'] = pserroc[label[0]]
        return prob_vector_sorted

    def predict(self, save=False, save_plot=False, show_plot=True, n_components=False, **kwargs):
        """ Runs default mode """
        M_best, AIC, BIC, models =  self.fit_model(autoparam=True, **kwargs)

        if self.autoparam or kwargs.get('autoparam', False):
            M_best = models[np.argmin(AIC)]
        else:
            if type(n_components)!=int:
                n_components = self.n_components
            M_best = models[n_components-1]
            self.model = M_best

        # Plot report
        if show_plot or save_plot:
            fig = plt.figure(figsize=(20, 6))
            ax = fig.add_subplot(121)
            if 'dens' in self.metric:
                lim = [0, 1]
                x = np.array(np.linspace(0,1,100)).reshape(-1,1)
                xplot = x[:,0]
            else:
                lim = [0, max(self.X)+np.std(self.X)]
                x = np.array(np.linspace(0,int(max(self.X)),100)).reshape(-1,1)
                xplot = x[:,0]
            logprob = M_best.score_samples(x)
            responsibilities = M_best.predict_proba(x)
            pdf = np.exp(logprob)
            pdf_individual = responsibilities * pdf[:, np.newaxis]

            ax.hist(self.X, 30, density=True, histtype='stepfilled', alpha=0.3)
            # ax.plot(xplot, pdf, '-k', label='Sample probability')
            # Iterate colors and labels in order
            my_cmap = sns.color_palette('mako', M_best.n_components).as_hex()
            order, corresp = self._label_sort_categories()
            for color, comp in zip(my_cmap, order):
                ax.plot(xplot, pdf_individual[:,corresp[comp]], c=color, linestyle='--', linewidth=5, label='P({})'.format(comp))
            #ax.plot(xplot, pdf_individual, c=my_cmap[0], linestyle='--', linewidth=5, label='Individual probabilities')
            ax.set_xlabel('{}'.format(self.metric))
            ax.set_ylabel('$p(x)$')
            ax.set_xlim(lim)
            ax.legend()

            # plot 2: AIC and BIC
            N = np.arange(1, kwargs.get('max_components', self.max_components))
            ax = fig.add_subplot(122)
            ax.plot(N, AIC, '-k', label='AIC')
            ax.plot(N, BIC, '--k', label='BIC')
            ax.set_xlabel('n. components')
            ax.set_ylabel('information criterion')
            ax.legend()

            if save_plot:
                plt.savefig(save_plot)
            if show_plot:
                plt.show()
            else:
                plt.close()

        # Results
        self.results = self.report(**kwargs)
        if save:
            self.results.to_excel(save)
        return self.results


class BayesianGaussianMixture(predictor):
    """
    Variational Bayesian estimation of a Gaussian mixture.
    This class allows to infer an approximate posterior distribution over the parameters of a Gaussian mixture distribution. The effective number of components can be inferred from the data.
    """
    def initialize(self, **kwargs):
        """ Accepts all arguments from https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html """
        return BGMM(n_components=kwargs.get('n_components', self.n_components),
                    covariance_type=kwargs.get('covariance_type', self.covariance_type),
                    init_params=kwargs.get('init_params', 'kmeans'), max_iter=kwargs.get('max_iter', 100),
                    n_init=kwargs.get('n_init', 1), random_state=kwargs.get('random_state', 42),
                    reg_covar=kwargs.get('reg_covar', 1e-06), tol=kwargs.get('tol', 0.001),
                    verbose=kwargs.get('verbose', 0), verbose_interval=kwargs.get('verbose_interval', 10),
                    warm_start=kwargs.get('warm_start', False),
                    weight_concentration_prior_type=kwargs.get('weight_concentration_prior_type', 'dirichlet_process'),
                    weight_concentration_prior=kwargs.get('weight_concentration_prior', None),
                    mean_precision_prior=kwargs.get('mean_precision_prior', None),
                    mean_prior=kwargs.get('mean_prior', None),
                    degrees_of_freedom_prior=kwargs.get('degrees_of_freedom_prior', None),
                    covariance_prior=kwargs.get('covariance_prior', None))

    def auto_parametrize(self, **kwargs):
        """
        Find the best fitting model without prior information
        """
        # Not AIC BIC defined 
        pass

    def fit_model(self, **kwargs):
        """ If autoparam, this will look for the best model. Run default otherwise """ 
        if self.autoparam or kwargs.get('autoparam', False):
            # not used in this method
            self.model, AIC, BIC, models = self.auto_parametrize(kwargs.get('criterion', 'AIC'), **kwargs)
            return self.model, AIC, BIC, models
        else:
            self.model = self.initialize(**kwargs).fit(self.X)
            return self.model

    def _label_sort_categories(self):
        rs = {}
        means = [i[0] for i in self.model.means_]
        n = len(means)
        if n in self._explabels and n==len(self._explabels[n]):
            class_correspondance = sorted([[m, self.model.predict([[m]])[0]] for m in means])
            ccc = set([v[1] for v in class_correspondance])  # Class correspondance checker required to avoid missing classes if very convoluted distribution
            if len(ccc)!=n:
                class_correspondance[0][1] = 1
                class_correspondance[1][1] = 2
                class_correspondance[2][1] = 0
            order = self._explabels[n]
        else:
            class_correspondance = [[round(x, 3), y] for x, y in sorted(zip(means, range(0,n)))]
            order =  ['comp{}'.format(c) for c in range(1, n+1)]
        class_correspondance = {c:m[1] for m, c in zip(class_correspondance, order)}
        return order, class_correspondance

    def report(self, **kwargs):
        """ Return dataframe of probabilities (results) """
        pm = {}
        rs = {}
        if 1==1:
            logprob = self.model.score_samples(self.X)
            responsibilities = self.model.predict_proba(self.X)
            pdf = np.exp(logprob)
            self.pdf = responsibilities * pdf[:, np.newaxis]
        else:
            self.pdf = self.model.predict_proba(self.X)
        self.labels = self.model.predict(self.X)

        c = 0
        # Prepare header
        header = ['identifier', 'left ann', 'right ann', 'strand', 'len', 'len_considered', self.metric, 'P(no insertions)', 'P(saturation)']
        order, corresp = self._label_sort_categories()
        pserroc = {v:k for k, v in corresp.items()} # reverse of corresp
        header += ['P({})'.format(i) for i in order]+['class']

        # Prob no insertions and saturation
        max_dens = max(self.model.means_)[0]
        pnis = probability_no_insertions(self.data, max_dens)
        psat = probability_saturation(self.data, max_dens)

        # populate array
        for ide, met, probs, label, pni, psa, used_len in zip(self.ides, self.X, self.pdf, self.labels, pnis, psat, self.data.metric['L']):
            st, en, strand = self.data.annotation.get(ide, ['NA', 'NA', 'NA'])
            try:
                l = abs(en-st+1)
            except:
                l = 'NA'
            prob_vector_sorted = [round(probs[corresp[e]],3) for e in order]
            rs[c] = [ide, st, en, strand, l, used_len, met[0], round(pni,3), round(psa,3)]+prob_vector_sorted+[pserroc[label]]
            c+=1
        # Return df
        rs = pd.DataFrame.from_dict(rs, orient='index')
        rs.columns = header
        return rs

    def value_predict(self, value, **kwargs):
        """ Return dataframe of probabilities (results) """
        X = [[value]]
        if not self.model:
            self.model = self.fit_model(**kwargs)
        logprob = self.model.score_samples(X)
        responsibilities = self.model.predict_proba(X)
        pdf = np.exp(logprob)
        pdf = responsibilities * pdf[:, np.newaxis]
        pdf = list(pdf[0]) # require for single pred
        label = self.model.predict(X)
        order, corresp = self._label_sort_categories()
        pserroc = {v:k for k, v in corresp.items()} # reverse of corresp

        prob_vector_sorted = {'P({})'.format(e):round(pdf[corresp[e]],3) for e in order}
        prob_vector_sorted['label'] = pserroc[label[0]]
        return prob_vector_sorted

    def predict(self, save=False, save_plot=False, show_plot=True, n_components=False, **kwargs):
        """ Runs default mode """
        fig = plt.figure(figsize=(6, 4))
        # plot 1: data + best-fit mixture
        ax = fig.add_subplot(111)

        if self.autoparam or kwargs.get('autoparam', False):
            # It will not enter here
            # M_best = models[np.argmin(AIC)]
            pass
        else:
            if type(n_components)!=int:
                n_components = self.n_components
            M_best =  self.fit_model(**kwargs)
            self.model = M_best

        # Plot report
        if show_plot or save_plot:
            if 'dens' in self.metric:
                lim = [0, 1]
                x = np.array(np.linspace(0,1,100)).reshape(-1,1)
                xplot = x[:,0]
            else:
                lim = [0, max(self.X)+np.std(self.X)]
                x = np.array(np.linspace(0,int(max(self.X)),100)).reshape(-1,1)
                xplot = x[:,0]
            logprob = M_best.score_samples(x)
            responsibilities = M_best.predict_proba(x)
            pdf = np.exp(logprob)
            pdf_individual = responsibilities * pdf[:, np.newaxis]

            ax.hist(self.X, 30, density=True, histtype='stepfilled', alpha=0.3)
            # ax.plot(xplot, pdf, '-k', label='Sample probability')
            # Iterate colors and labels in order
            my_cmap = sns.color_palette('mako', M_best.n_components).as_hex()
            order, corresp = self._label_sort_categories()
            for color, comp in zip(my_cmap, order):
                ax.plot(xplot, pdf_individual[:,corresp[comp]], c=color, linestyle='--', linewidth=5, label='P({})'.format(comp))
            #ax.plot(xplot, pdf_individual, c=my_cmap[0], linestyle='--', linewidth=5, label='Individual probabilities')
            ax.set_xlabel('{}'.format(self.metric))
            ax.set_ylabel('$p(x)$')
            ax.set_xlim(lim)
            ax.legend()

            if save_plot:
                plt.savefig(save_plot)
            if show_plot:
                plt.show()
            else:
                plt.close()

        # Results
        self.results = self.report(**kwargs)
        if save:
            self.results.to_excel(save)
        return self.results


class HiddenMarkov(predictor):
    """
    Re-implementation of the HMM approach presented by Michael A DeJesus et al (2013)
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3854130/
    """

    def _define_mu_for_genes(self, d, genes):
        reads = []
        for g in genes:
            v = d.annotation[g]
            for r in d.zreads[v[0]-1:v[1]]:
                reads.append(r)
        return np.mean(reads), np.std(reads)

    def _define_mu_and_labels(self, mode=1, expected_labels=None, mu=None):
        """
        Define the expected mu for each class distribution,
        mode=0 : reference from paper, it seems to return proper results only for TA-specific transposon
        mode=1 : custom based on gold set of NE genes to define mu of N
        mode=2 : mode 1 with extra class as NE+std(NE)
        """

        if expected_labels and mu:
            label = expected_labels
            mu = mu
            L = 1.0/mu
            N = len(L)
        elif mode==0:
            # Original paramete
            reads_nz = sorted(self.data.reads)
            mean_r = np.average(reads_nz[:int(0.95 * len(reads_nz))]) # remove 5% percentile
            mu = np.array([1/0.99, 0.01 * mean_r + 2,  mean_r, mean_r*5.0])
            L = 1.0/mu
            N = 4
            label= {0:"E", 1:"F", 2:"N",3:"A"}
        elif mode==1:
            _ne_mu, _ne_std = self._define_mu_for_genes(self.data, self.data.goldNE)
            mu = np.array([1, _ne_mu/2, _ne_mu])
            L = 1.0/mu
            N = 3
            label= {0:"E", 1:"F", 2:"N"}
        elif mode==2:
            _ne_mu, _ne_std = self._define_mu_for_genes(self.data, self.data.goldNE)
            mu = np.array([1, _ne_mu/2, _ne_mu, _ne_mu+_ne_std])
            L = 1.0/mu
            N = 4
            label= {0:"E", 1:"F", 2:"N", 3:"A"}
        else:
            sys.exit('No mode selected')
        return mu, L, N, label

    def _calculate_pins(self, observations, **kwargs):
        """
        Function to calculate probabilities of initial states
        This code comes from the original version by Michael A DeJesus et al (2013),
        it basically discards to count 0s if separated more than 10 positions,
        10 seems to be completely arbitrary (it can be chaged through keyword arguments
        """
        space = kwargs.get('space', 10)
        non_ess_obs = []
        tmp = []
        for o in observations:
            if o>=1:
                if len(tmp)<space:
                    non_ess_obs.extend(tmp)
                non_ess_obs.append(o)
                tmp = []
            else:
                tmp.append(o)
        return sum([1 for o in non_ess_obs if o>=1])/float(len(non_ess_obs))

    def viterbi(self, A, B, PI, O, **kwargs):
        """
        A: transition probabilities
        B: emission probability distributions
        PI: initial state probabilities
        O: observations
        """
        O = O+1   # Require to avoid having a null initial state probability

        #TODO this functions is quite time consuming, consider refactorization
        scaling = kwargs.get('scaling', True)
        discrete = kwargs.get('discrete', False)

        n_states=len(B)
        T = len(O)
        delta = np.zeros((n_states, T))
        Q = np.zeros((n_states, T))

        # Init
        b_o = [B[i](O[0]) for i in range(n_states)]
        if scaling:
            delta[:,0] = np.log(PI) + np.log(b_o)
        # Populate
        for t in range(1, T):
            b_o = [B[i](O[t]) for i in range(n_states)]
            #nus = delta[:, t-1] + np.log(A)
            nus = delta[:, t-1] + A
            delta[:,t] = nus.max(1) + np.log(b_o)
            Q[:,t] = nus.argmax(1)
        Q_opt = [np.argmax(delta[:,T-1])]
        for t in range(T-2, -1, -1):
            Q_opt.insert(0, Q[int(Q_opt[0]),t+1])
        return((Q_opt, delta, Q))

    def hash_genes(self):
        if not self.data._one_base_annotation_:
            self.data.annotation_per_base()
        return self.data._one_base_annotation_

    def VarR(self, n,p):
        """ VarR_n =  (pi^2)/(6*ln(1/p)^2) + 1/12 + r2(n) + E2(n) (Schilling, 1990)"""
        r2 = 0.00006
        E2 = 0.01
        A = pow(pi,2.)/(6*pow(log(1.0/p),2.0))
        V = A + 1/12.0 + r2 + E2
        return V

    def ExpectedRuns(self, n,p):
        """ER_n =  log(1/p)(nq) + gamma/ln(1/p) -1/2 + r1(n) + E1(n) (Schilling, 1990)"""
        q = 1-p
        gamma = 0.5772156649015328606     # Euler-Mascheroni constant
        r1 = 0.000016
        E1 = 0.01
        A = log(n*q,1.0/p)
        B = gamma/log(1.0/p)
        ER = A + B -0.5 + r1 + E1
        return ER

    def evaluate(self):
        """ Assign a label per position in the genome """
        #init
        hast = self.hash_genes()   # dict as position:[annotations in that base]
        state2count = {k:0 for k in range(self.N)}
        self.gene_profile = {}
        #iterate
        for pos, anns in hast.items():
            if len(anns)==0:
                anns = ['intergenic']
            for orf in anns:
                if orf in self.gene_profile:
                    self.gene_profile[orf][1].append(pos)  # positions
                    self.gene_profile[orf][2].append(self.data.zreads[pos-1])  # read values (-1 as zreads is 0 based but not hast
                    self.gene_profile[orf][3].append(self.label.get(int(self.Q_opt[pos-1]), 'Unknown State')) # State label
                else:
                    self.gene_profile[orf] = [orf, [pos], [self.data.zreads[pos-1]], [self.label.get(int(self.Q_opt[pos-1]), 'Unknown State')]]
        self.gene_profile = {k:[v[0], np.array(v[1]), np.array(v[2]), np.array(v[3])] for k, v in self.gene_profile.items()}

    def label_genes(self, ND=True, other_results=None):
        """
        Label genes based on the state labels of a genes
        nd : Ignore plausible domain essentials, ignoring statistically significant stretches of ES states. (Default=True)
        """
        # Generate results and plot evalutation
        if other_results:
            iterator = other_results.items()
        else:
            iterator = self.gene_profile.items()
        results = {}
        state_labels = [self.label[i] for i in range(self.N)]
        header  = ['identifier', 'left ann', 'right ann', 'strand', 'len', 'len_considered', 'density [Nr Ins/L]', 'avg reads>0']+state_labels+['state']
        c = 0
        for gene, profile in iterator:
            genename, positions, reads, states = profile
            st, en, strand = self.data.annotation.get(gene, ['NA', 'NA', 'NA'])
            try:
                l = abs(en-st+1)
            except:
                l = 'NA'
            len_considered = len(reads)   # number of states (positions, reads and states share the same lengths)
            nzreads = reads[reads>0]
            dens = len(nzreads)/len_considered
            if len(nzreads)>0:
                avg_read_nz = np.average(nzreads)
            else:
                avg_read_nz = 0

            # Count states
            statedist = Counter(states)
            if not ND:
                if dens==0:
                    dens = 1/len(reads)
                E = self.ExpectedRuns(len_considered, 1.0-dens)
                V = self.VarR(len_considered, 1.0-dens)
                if statedist[0]==len_considered or statedist[0]>=int(E+(3*sqrt(V))):
                    S = self.label[0]   # Assign first class
                else:
                    S = max([(statedist.get(s, 0), s) for s in list(self.label.values())])[1]
            else:
                S = max([(statedist.get(s, 0), s) for s in list(self.label.values())])[1]
            # Append results
            results[c] = [genename, st, en, strand, l, len_considered, dens, avg_read_nz]+[statedist.get(state, 0) for state in state_labels]+[S]
            c+=1
        results = pd.DataFrame.from_dict(results, orient='index')
        results.columns = header
        return results

    def predict(self,
                mode=1, expected_labels=None, mu=None,
                save=False, save_plot=False, show_plot=True,
                transition={}, _nd=True,
                **kwargs):
        """
        Takes R (reads) as metric, if other metric is required it
        can be passed with the 'metric' keyword as argument.

        expected_labels passed as dictionary:
        E - essential, F - fitness, N - non-essential, A - advantageous
        default --> {0:"E", 1:"F", 2:"N", 3:"A"}

        prior allows to select different expected mu parameters per class in the
        shape: {0:0.0, 1:0.1, 2:0.5, 3:2.5} # it has to match expected labels

        transition allows to select different expected transition probabilites per class in the
        shape: {0:0.0, 1:0.1, 2:0.5, 3:2.5} # it has to match expected labels
        They are calculated using the default from Michael A DeJesus et al (2013)

        """

        # Define metric
        self.metric  = kwargs.get('metric', 'R')
        if self.metric == 'R':
            self.observations = self.data.zreads
        else:
            sys.exit('Metric {} not implemented'.format(self.metric))
        non_zero_observations = sorted(self.observations[self.observations>0])
        size = len(non_zero_observations)

        # Define class mu
        self.mu, self.L, self.N, self.label = self._define_mu_and_labels(mode=mode, expected_labels=expected_labels, mu=mu)

        # Define emission probabilities
        self.emissions = [stats.geom(i).pmf for i in self.L]

        # Define pins and transition probabilities
        self.pins = self._calculate_pins(self.observations, **kwargs)
        self.pins_obs = sum([1 for o in self.observations if o>=2])/float(len(self.observations))
        if len(transition)==0 or transition=='default':
            pnon = 1.0-self.pins
            pnon_obs = 1.0-self.pins_obs
            for r in range(100):
                if pnon**r < 0.01: break       #TODO set 0.01 as significance value?
            self.transitions = np.zeros((self.N, self.N))
            _a = log1p(-self.emissions[int(self.N/2)](1)**r)
            _b = r*log(self.emissions[int(self.N/2)](1)) + log(1.0/3)
            for i in range(self.N):
                self.transitions[i] = [_b]*self.N
                self.transitions[i][i] = _a
        else:
            assert len(transition)==len(expected_labels)
            sys.exit('Custom transition not implemented') #TODO set custom transition

        # Define initial state
        self.PI = np.zeros(self.N)
        self.PI[0] = 0.7
        self.PI[1:] = 0.3/(self.N-1)    #TODO check definition of initial states

        # Run viterbi
        self.Q_opt, self.delta, self.Q = self.viterbi(A=self.transitions, B=self.emissions, PI=self.PI, O=self.observations, **kwargs)  # scaling = True, discrete = False as kwargs

        # Profile genes
        self.evaluate()    # adds self.gene_profile
        self.results = self.label_genes(ND=_nd)

        if save:
            self.results.to_excel(save)

        # Plot
        if show_plot or save_plot:
            sns.scatterplot(x='density [Nr Ins/L]', y='avg reads>0', hue='state', style='state', palette='mako', data=self.results)
            plt.legend()
            plt.title('HMM gene classification')
            if save_plot:
                plt.tight_layout()
                plt.savefig(save_plot)
            if show_plot:
                plt.show()
            else:
                plt.close()

        return self.results
