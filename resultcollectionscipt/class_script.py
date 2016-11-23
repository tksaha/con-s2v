#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import re
import os
import sys 

d = {}

method_names = []
metric_names = ['Precision', 'Recall', 'F1-score', 'Accuracy', 'Kappa']


# read method names
for line in open(sys.argv[1]):
	if "Report" in line:
		pre, _, post = line.strip().partition("(")
		method_name, _, post = post.partition(")")
		if method_name not in method_names:
			method_names.append(method_name)

for method in method_names:
	for metric in metric_names:
		d[(method, metric)] = []


with open(sys.argv[1], 'r') as f:
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

df[df.columns[0:7]].to_csv("temp_class.csv", index=False)