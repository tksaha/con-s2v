#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys 
import pandas as pd
import re

d = {}

method_names = []
metric_names = ['HomoGeneity', 'Completeness', 'v_measure', 'Adjusted Mutual Info Score']



# read method names
for line in open(sys.argv[1]):
	if "####" in line and "Iteration" not in line:
		pre, _, post = line.strip().partition("#######")
		method_name, _, post = post.partition("#############")
		if method_name not in method_names:
			method_names.append(method_name)

for method in method_names:
	for metric in metric_names:
		d[(method, metric)] = []

with open(sys.argv[1]) as f:
	f = f.read()
	#f = f.replace("Completeness", "Completeness:")
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

df[df.columns[0:7]].to_csv("temp_clust.csv", index=False)
