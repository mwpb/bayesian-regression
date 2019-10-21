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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
# #!pip3 install jupytext

df = pd.read_csv('./poverty.csv')

df.head()

X = df['PovPct']
y = df['Brth15to17']

plt.scatter(X, y)

basic_model = pm.Model()
with basic_model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    mu = alpha + beta*X
    
    y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y) # observed apparently makes this the likelihood
    # presumably for glm one replaces normal with the appropriate residual.

map_estimate = pm.find_MAP(model=basic_model)
map_estimate

with basic_model:
    trace = pm.sample(5000)

pm.traceplot(trace)

# +
# https://docs.pymc.io/notebooks/GLM-hierarchical.html
