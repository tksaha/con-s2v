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



	def __generateSummary(self, dumpingfactor):
		"""
		pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, 
		nstart=None, weight='weight', dangling=None)
		G = nx.Digraph(nx.path_graph(4))
		pr = nx.pagerank (G, alpha=0.9)
		Returns dictionary of nodes with pagerank as value 

		PageRank function always make a matrix right stochastic i.e. a real square matrix, 
		with each row summing to 1. 
		"""
		pageRankDict = nx.pagerank(self.nx_G, alpha=dumpingfactor)
		self.pagerankedNodes = sorted(pageRankDict.items(), key=operator.itemgetter(1), reverse=True) 


	def __getDiverseSummary(self, dumpingfactor, lambda_val):
		"""
		Generate Diverse Summary. It dumps summary 
		sentences based on an optimization function which 
	    trades off between ranking and diversity. 
		S is the set of examples already sampled. 
		Rank is returned by page rank. 
		Cos is cosine similarity 
		Implementing min {I\S} (lambda* rank  + max [j \in S] {( 1 - lambda )*maxcos} )
		"""
		
		candidates = []
		id_value_list = [] 

		position = 1
		for key, page_rankvalue in self.pagerankedNodes:
			
			rank = float(position)/len(self.pagerankedNodes)
			cos_val = 0.0
			fn_val = (1.0 - lambda_val)*cos_val
			# id, rank, cos, fn_value, pagerank_value 
			id_value_list.append((key, rank, cos_val, fn_val, pagerank_value))
			position = position + 1

		cur_pos = 0
		while len(candidates) < len(self.pagerankedNodes):
			if len(candidates) == 0:
				candidates.append((id_value_list[0][0], id_value_list[0][4]))
				cur_pos = cur_pos + 1
			else:
				# Go through the list and refine cos 
				for pos in range(cur_pos, len(id_value_list)):
					current_cos = id_value_list[pos][2]
					if current_cos < self.nx_G[id_value_list[current_pos-1][0]][id_value_list[pos][0]]['weight']:
						current_cos = self.nx_G[id_value_list[current_pos-1][0]][id_value_list[pos][0]]['weight']
						value = lambda_val * id_value_list[pos][1] + (1.0 - lambda_val)*current_cos
						id_value_list[pos][2] = current_cos
						id_value_list[pos][3] = value 


				sorted_list = sorted(id_value_list[cur_pos:], key = lambda tup: tup[3])
				candidate_key = sorted_list[0][0]
				candidates.append((candidate_key, sorted_list[0][4]))

				#swap 
				pos = 0
				for ky, _, _, _ in id_value_list:
					if candidate_key == ky:
						break 
					else:
						pos = pos+1

				if pos == cur_pos :
					cur_pos = cur_pos + 1
				else:
					temp = id_value_list[cur_pos]
					id_value_list[cur_pos] = id_value_list[pos]
					id_value_list[pos] = temp 
					cur_pos = cur_pos + 1

		for key, value in candidates:
			yield key, value  


	def getSummary(self, dumpingfactor, diversity, lambda_val = 1.0):
		"""
		This will return a summary sentence ID sorted by rank. 
		Implemented as a generator structure  
		"""
		self.__generateSummary(dumpingfactor)

		if diversity == False: 
			for key,value in self.pagerankedNodes:
				yield key, value 
		else:
			self.__getDiverseSummary(dumpingfactor, lambda_val)
	

	

