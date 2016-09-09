#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import random


class Node2Vec: 

	def __init__(self, *args, **kwargs):
		"""
		"""
		self.dimensions = kwargs['dimension'] 
		self.window_size = kwargs['window_size']
		args.cpu_count = kwargs['cpu_count']
		self.outputfile = kwargs['outputfile']
		self.num_walks = kwargs['num_walks']
		self.walk_length = kwargs['walk_length']
		self.p = kwargs['p']
		self.q = kwargs['q']


	def learn_embeddings(walks):
		"""
		Learn embeddings by optimizing the Skipgram 
		objective using SGD.
		"""
		walks = [map(str, walk) for walk in walks]
		model = Word2Vec(walks, size=self.dimension,\
			 window=self.window_size, min_count=0,\
			 workers=self.cpu_count)
		model.save_word2vec_format(self.outputfile)
	
		return model 

	def get_representation(nx_G):
		"""
		Pipeline for representational learning for all nodes in a graph.
		"""

		n2vWalk= Node2VecWalk(nx_G, False, self.p, self.q)
		n2vWalk.preprocess_transition_probs()
		walks = n2vWalk.simulate_walks(self.num_walks, self.walk_length)
		return learn_embeddings(walks)