import math
import scipy.stats
import numpy as np
import pandas
from collections import defaultdict

df = list(pandas.read_csv("poverty.csv").itertuples(index=False))

def probability(mean, intercept, std):
	p = 0
	for row in df:
		x, y = row[0], row[1]
		# print(mean, intercept, std)
		p += scipy.stats.norm(intercept+mean*x, std).logpdf(y)
		# print('p = '+str(p))
	return p

def next(intercept, mean , std):
	new_mean = mean + np.random.normal(0, 0.1, 1)[0]
	new_intercept = intercept + np.random.normal(0, 0.1, 1)[0]
	new_std = std + abs(np.random.normal(0, 0.1, 1)[0])
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
	return mean, intercept, std

mean, intercept, std = 0, 0, 1

dist_mean = defaultdict(int)
dist_std = defaultdict(int)
for i in range(200):
	mean, intercept, std = next(mean, intercept, std)
	dist_std[round(std, 1)] += 1
	print(dist_std)