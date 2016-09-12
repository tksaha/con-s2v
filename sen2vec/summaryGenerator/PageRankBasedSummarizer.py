#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 

import operator 
from summaryGenerator.Summarizer import Summarizer


class PageRankBasedSummarizer(Summarizer):
	"""
	Implements PageRank summarizer using graph as a data structure. 
	We use networkx as the graph data structure. 
	"""
	def __init__(self, *args, **kwargs):
		"""
		"""
		self.nx_G = kwargs['nx_G']
		self.pagerankedNodes = {}
		# for a, b, data in sorted(self.nx_G.edges(data=True), key= lambda abdata: (abdata[0],abdata[1])):
		# 	print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))



	def _generateSummary(self, dumpingfactor):
		"""
		pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
		G = nx.Digraph(nx.path_graph(4))
		pr = nx.pagerank (G, alpha=0.9)
		Returns dictionary of nodes with pagerank as value 
		"""
		pageRankDict = nx.pagerank(self.nx_G, alpha=dumpingfactor)
		self.pagerankedNodes = sorted(pageRankDict.items(), key=operator.itemgetter(1), reverse=True) 


	def getSummary(self, dumpingfactor):
		"""
		This will return a summary sentence ID sorted by rank. 
		Implemented as a generator structure  
		"""
		self._generateSummary(dumpingfactor)

		for key,value in self.pagerankedNodes:
			yield key, value 

