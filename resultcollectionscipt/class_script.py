#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import re
import os

d = {}

method_names = ['p2v', 'n2v', 'n2v_int', 'n2v_retrofit', 'Iter_unw', 'Iter_w', 'regsen2vec_w', 'regsen2vec_unw', 'dictreg_w_wordnet', 'dictreg_unw_wordnet']
metric_names = ['Precision', 'Recall', 'F1-score', 'Accuracy', 'Kappa']

for method in method_names:
	for metric in metric_names:
		d[(method, metric)] = []

with open('./Data/reuter_testresults_CLASS.txt', 'r') as f:
	f = f.read()
	runs = re.compile("###### Iteration.*######").split(f)
	
	for r, run in enumerate(runs):
		methods = re.compile("######Classification Report.*######").split(run)
		methods = [method for method in methods if "Kappa" in method]
		for m, method in enumerate(methods):
			values = method.split("avg / total")[1].split(os.linesep)[0].split()[0:3]
			values += [method.split("######Accuracy Score######")[1].split(os.linesep)[1]]
			values += [method.split("######Cohen's Kappa######")[1].split(os.linesep)[1]]
			
			for v, metric in enumerate(metric_names):
				d[(method_names[m], metric)] += [values[v]]

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
