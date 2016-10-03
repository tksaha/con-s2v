#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import random
from log_manager.log_config import Logger 
from cachetools import LRUCache

class Node2VecWalk:
	def __init__(self, nx_G, is_directed, p, q):
		"""
		"""
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		"""
		Simulate a random walk starting from start node.
		"""
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def node2vec_walk_no_precalc(self, walk_length, start_node, cache):
		"""
		Simulate a random walk starting from start node.
		"""
		G = self.G
		alias_nodes = self.alias_nodes
		walk = [start_node]
		

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					J, q = np.zeros(4), np.zeros(4)
					try:
						J, q = cache[(prev,cur)]
						Logger.logr.info("Using from cache")
					except:
						J, q = self.get_alias_edge(prev,cur)
						cache[(prev, cur)] = (J, q)
						#Logger.logr.info("Updating cache")

					next_ = cur_nbrs[self.alias_draw(J, q)]
					walk.append(next_)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length, precalc=True):
		"""
		Repeatedly simulate random walks from each node.
		64 * 2^10= 
		"""
		if precalc ==True: 
		   self.preprocess_transition_probs()
		else: 
		   self.preprocess_only_node_probs()

		G = self.G
		walks = []
		nodes = list(G.nodes())
		Logger.logr.info('Walk iteration:')
		cache =  LRUCache(maxsize= (64 * 1024))

		for walk_iter in range(num_walks):
			Logger.logr.info("%s %s %s" %(str(walk_iter+1), '/', str(num_walks)))
			random.shuffle(nodes)
			for node in nodes:
				if precalc == True:
					walk = self.node2vec_walk(walk_length=walk_length, start_node=node)
				else:
					walk = self.node2vec_walk_no_precalc(walk_length=walk_length, start_node=node, cache=cache)
				yield walk
		
		#return walks


	def get_alias_edge(self, src, dst):
		"""
		Get the alias edge setup lists for a given edge.
		"""

		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return self.alias_setup(normalized_probs)

	

	def preprocess_only_node_probs(self):
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = self.alias_setup(normalized_probs)
			Logger.logr.info("calculating transition probability for %i" %node)
		self.alias_nodes = alias_nodes


	def preprocess_transition_probs(self):
		"""
		Preprocessing of transition probabilities for guiding the random walks.
		"""

		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = self.alias_setup(normalized_probs)
			Logger.logr.info("calculating transition probability for %i" %node)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


	def alias_setup(self, probs):
		"""
		Compute utility lists for non-uniform sampling from discrete distributions.
		Refer to https://hips.seas.harvard.edu/blog/2013/03/03/
		the-alias-method-efficient-sampling-with-many-discrete-outcomes/
		for details
		"""

		K = len(probs)
		q = np.zeros(K)
		J = np.zeros(K, dtype=np.int)

		smaller = []
		larger = []
		for kk, prob in enumerate(probs):
		    q[kk] = K*prob
		    if q[kk] < 1.0:
		        smaller.append(kk)
		    else:
		        larger.append(kk)

		while len(smaller) > 0 and len(larger) > 0:
		    small = smaller.pop()
		    large = larger.pop()

		    J[small] = large
		    q[large] = q[large] + q[small] - 1.0
		    if q[large] < 1.0:
		        smaller.append(large)
		    else:
		        larger.append(large)

		return J, q

	def alias_draw(self, J, q):
		"""
		Draw sample from a non-uniform discrete distribution using alias sampling.
		"""
		K = len(J)

		kk = int(np.floor(np.random.rand()*K))
		if np.random.rand() < q[kk]:
		    return kk
		else:
		    return J[kk]
