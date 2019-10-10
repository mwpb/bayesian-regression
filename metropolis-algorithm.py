# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import math
import scipy.stats
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pylab as plt

df = pd.read_csv("poverty.csv")

df.columns


class MetropolisRegression:
    
    cache = {}
    freqs = {
        'mean':defaultdict(int),
        'intercept':defaultdict(int),
        'std':defaultdict(int)
    }
    
    def __init__(self, X, y, residual_dist, mean=0, intercept=0, std=1, movement = 0.5):
        self.mean, self.intercept, self.std = mean, intercept, std
        self.residual_dist = residual_dist
        self.movement = movement
        self.X, self.y = X, y
        
    def probability(self, mean, intercept, std):
        if std == 0:
            return 0
        mean, intercept, std = round(mean, 1), round(intercept, 1), round(std, 1)
        if (mean, intercept, std) not in self.cache:
            p = 0
            for i in range(len(self.X.index)):
                x, y = self.X[i], self.y[i]
                p += self.residual_dist(intercept+mean*x, std).logpdf(y)
            self.cache[mean, intercept, std] = p
        return self.cache[mean, intercept, std]
    
    def should_accept(self, new_mean, new_intercept, new_std):
        old_log_prob = self.probability(self.mean, self.intercept, self.std)
        new_log_prob = self.probability(new_mean, new_intercept, new_std)
        log_acceptance_probability = new_log_prob - old_log_prob
        if log_acceptance_probability >= 0:
            return True
        else:
            u = np.random.uniform(0, 1, 1)
            if math.log(u) <= log_acceptance_probability:
                return True
        return False
    
    def next_sample(self):
        new_mean = self.mean + np.random.normal(0, self.movement, 1)[0]
        new_intercept = self.intercept + np.random.normal(0, self.movement, 1)[0]
        new_std = self.std + np.random.normal(0, self.movement, 1)[0]
        
        if self.should_accept(new_mean, new_intercept, new_std):
            self.mean, self.intercept, self.std = new_mean, new_intercept, new_std
        
        self.freqs['mean'][round(self.mean, 1)] += 1
        self.freqs['intercept'][round(self.intercept, 1)] += 1
        self.freqs['std'][round(self.std, 1)] += 1
        
    def run(self, n):
        for i in range(n):
            self.next_sample()
            
    def plot_mean(self):
        lists = sorted(self.freqs['mean'].items())
        x, y = zip(*lists)
        plt.xlim = (1.5, 1.7)
        plt.plot(x, y)
        plt.show()
        
    def plot_intercept(self):
        lists = sorted(self.freqs['intercept'].items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.show()
        
    def plot_std(self):
        lists = sorted(self.freqs['std'].items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.show()


m = MetropolisRegression(df['PovPct'], df['Brth15to17'], scipy.stats.norm, movement = 0.5)

# %%time
m.run(1000)

m.plot_mean()
m.plot_intercept()
m.plot_std()

df.plot(kind='scatter', x = 'PovPct', y = 'Brth15to17')

import statsmodels.api as sm

X = sm.add_constant( df['PovPct'])
model = sm.OLS(df['Brth15to17'],X)
results = model.fit()
print(results.summary())
















