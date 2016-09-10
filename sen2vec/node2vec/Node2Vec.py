#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec


from node2vec.Node2VecWalk import Node2VecWalk


class Node2Vec: 

	def __init__(self, *args, **kwargs):
		"""
		"""
		self.dimension = kwargs['dimension'] 
		self.window_size = kwargs['window_size']
		self.cpu_count = kwargs['cpu_count']
		self.outputfile = kwargs['outputfile']
		self.num_walks = kwargs['num_walks']
		self.walk_length = kwargs['walk_length']
		self.p = kwargs['p']
		self.q = kwargs['q']


	def learn_embeddings(self, walks):
		"""
		Learn embeddings by optimizing the Skipgram 
		objective using SGD.
		"""
		walks = [list(map(str, walk)) for walk in walks]		
		model = Word2Vec(walks, size=self.dimension,\
			 window=self.window_size, min_count=0,\
			 workers=self.cpu_count)
		model.save_word2vec_format(self.outputfile, binary=False)
	
		return model 
		

	def get_representation(self, nx_G):
		"""
		Pipeline for representational learning for all nodes in a graph.
		"""

		n2vWalk= Node2VecWalk(nx_G, False, self.p, self.q)
		n2vWalk.preprocess_transition_probs()
		walks = n2vWalk.simulate_walks(self.num_walks, self.walk_length)
		return self.learn_embeddings(walks)