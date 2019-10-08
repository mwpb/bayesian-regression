import math
import scipy.stats
import numpy as np
import pandas
from collections import defaultdict

df = list(pandas.read_csv("poverty.csv").itertuples(index=False))

cache = {}

def probability(mean, intercept, std):
    if (mean, intercept, std) not in cache:
        p = 0
        for row in df:
            x, y = row[0], row[1]
            p += scipy.stats.norm(intercept+mean*x, std).logpdf(y)
        cache[mean, intercept, std] = p
    return cache[mean, intercept, std]

def next(intercept, mean , std):
    new_mean = mean + np.random.normal(0, 0.1, 1)[0]
    new_intercept = intercept + np.random.normal(0, 0.1, 1)[0]
    new_std = max(0, std + np.random.normal(0, 0.1, 1)[0])
    # print('new:')
    # print(new_mean, new_intercept, new_std)
    log_a = probability(new_mean, new_intercept, new_std) - probability(mean, intercept, std)
    if log_a >= 0:
        mean, intercept, std = new_mean, new_intercept, new_std
    else:
        u = np.random.uniform(0, 1, 1)
        if math.log(u) <= log_a:
            mean, intercept, std = new_mean, new_intercept, new_std
    print(mean, intercept, std)
    return round(mean, 1), round(intercept, 1), round(std, 1)

mean, intercept, std = 0, 0, 1

dist_mean = defaultdict(int)
dist_std = defaultdict(int)
for i in range(200):
    mean, intercept, std = next(mean, intercept, std)
    dist_mean[round(mean, 1)] += 1
    print(dist_mean)

out_df = pandas.from_dict(dist_mean)