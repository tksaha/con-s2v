#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import re

d = {}

method_names = ['p2v', 'n2v', 'n2v_int', 'n2v_retrofit', 'Iter_unw', 'Iter_w', 'regsen2vec_w', 'regsen2vec_unw', 'dictreg_w_wordnet', 'dictreg_unw_wordnet']
metric_names = ['HomoGeneity', 'Completeness', 'v_measure', 'Adjusted Mutual Info Score']

for method in method_names:
	for metric in metric_names:
		d[(method, metric)] = []

with open('../Data/news_testresults_CLUST.txt', 'r') as f:
	f = f.read()
	f = f.replace("Completeness", "Completeness:")
	runs = re.compile("###### Iteration.*######").split(f)
	
	for r, run in enumerate(runs):
		methods = re.compile("#######.*#############").split(run)
		methods = [method for method in methods if "HomoGeneity" in method]
		for m, method in enumerate(methods):
			for metric in metric_names:
				value = method.split(metric+":")[1].split(" ")[0]
				d[(method_names[m], metric)] += [value]

df = pd.DataFrame(columns = ['metric', 'method', 'run1', 'run2', 'run3', 'run4' , 'run5'])
for key, value in d.items():
	df.loc[df.shape[0]] = [key[1], key[0], value[0], value[1], value[2], value[3], value[4]]

df['method_ordered'] = pd.Categorical(
	df['method'], 
	categories=method_names, 
	ordered=True
)

df = df.sort_values(by=['metric', 'method_ordered'])

df[df.columns[0:7]].to_csv("temp.csv", index=False)
