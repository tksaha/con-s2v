#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 
from log_manager.log_config import Logger 
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

	def __printTuples(self, tup):
		Logger.logr.info("%i %lf %lf %lf %lf"%(tup[0], tup[1], tup[2], tup[3], tup[4]))


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
		
		#Logger.logr.info("[Diverse]Generating diverse summary with lambda=%lf"%lambda_val)
		candidates = []
		id_value_list = [] 

		position = 1
		for key, page_rankvalue in self.pagerankedNodes:	
			rank = float(position)/len(self.pagerankedNodes)
			cos_val = 0.0
			fn_val = (lambda_val * rank)  + ((1.0 - lambda_val)*cos_val)
			id_value_list.append((key, rank, cos_val, fn_val, page_rankvalue))
			#Logger.logr.info("Appending key=%i, rank=%lf, cos=%lf,"\
			#" fn_val=%lf, page=%lf" %(key, rank, cos_val, fn_val, page_rankvalue))
			position = position + 1

		cur_pos = 0
		while len(candidates) < len(self.pagerankedNodes):
			#Logger.logr.info("Total candidate=%i, total nodes=%i"%(len(candidates), len(self.pagerankedNodes)))
			if len(candidates) == 0:
				candidates.append((id_value_list[0][0], id_value_list[0][4]))
				cur_pos = cur_pos + 1
			else:
				# Go through the list and refine cos 
				for pos in range(cur_pos, len(id_value_list)):
					current_cos = id_value_list[pos][2]
					similarity = 0.0
					try:
						similarity = self.nx_G[id_value_list[cur_pos-1][0]][id_value_list[pos][0]]['weight']
						#Logger.logr.info("Getting value %lf"%similarity)
					except: 
						pass 
					if current_cos < similarity:
						current_cos = similarity
						value = (lambda_val * id_value_list[pos][1]) + ((1.0 - lambda_val)*current_cos)
						#self.__printTuples(id_value_list[pos])
						tup = (id_value_list[pos][0], id_value_list[pos][1], current_cos, value, id_value_list[pos][4])
						id_value_list[pos] = tup 
						#self.__printTuples(id_value_list[pos])


				sorted_list = sorted(id_value_list[cur_pos:], key = lambda tup: tup[3])
				candidate_key = sorted_list[0][0]
				candidates.append((candidate_key, sorted_list[0][4]))
				#swap 
				pos = 0
				for id_vals in id_value_list:
					ky, _, _, _ = id_vals[0],id_vals[1],id_vals[2],id_vals[3] 
					if candidate_key == ky:
						break 
					else:
						pos = pos+1
				if pos == cur_pos :
					cur_pos = cur_pos + 1
				else:
					#Logger.logr.info("Exchanging ............%i with %i" %(cur_pos, pos))
					temp = id_value_list[cur_pos]
					id_value_list[cur_pos] = id_value_list[pos]
					id_value_list[pos] = temp 
					cur_pos = cur_pos + 1

		return candidates


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
			#Logger.logr.info("Generating diverse summary with lambda=%lf"%lambda_val)
			for candidate in self.__getDiverseSummary(dumpingfactor, lambda_val):
				#Logger.logr.info("[Diverse]Generating diverse summary with lambda=%lf"%lambda_val)
				yield candidate[0], candidate[1]
	

	

