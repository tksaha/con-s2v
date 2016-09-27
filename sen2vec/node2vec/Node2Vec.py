#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec
import subprocess 
from log_manager.log_config import Logger 

from node2vec.Node2VecWalk import Node2VecWalk
from word2vec.WordDoc2Vec import WordDoc2Vec


class Node2Vec: 

	def __init__(self, *args, **kwargs):
		"""
		"""
		self.dimension = kwargs['dimension'] 
		self.window_size = kwargs['window_size']
		self.outputfile = kwargs['outputfile']
		self.num_walks = kwargs['num_walks']
		self.walk_length = kwargs['walk_length']
		self.dataDir = os.environment['DATADIR']
		self.p = kwargs['p']
		self.q = kwargs['q']


	def learnEmbeddings(self, walkInput, initFromFile, initFile):
		"""
		Learn embeddings by optimizing the Skipgram 
		objective using SGD. [GENSIM]
		"""
		Logger.logr.info("Starting the Walk")
		wordDoc2Vec = WordDoc2Vec()
		wPDict = wordDoc2Vec.buildWordDoc2VecParamDict()
		wPDict["cbow"] = 0 
		wPDict["sentence-vectors"] = 0
		wPDict["min-count"] = 0
		wPDict["train"] = walkInput
		wPDict["output"] = self.outputfile
		wPDict["size"]= 600
		args = []
		if initFromFile==True:
			wPDict["init"] = initFile
			args = wPDict.buildArgListforW2V(wPDict)
		else:
			args = wPDict.buildArgListforW2VWithInit(wPDict)
		process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = process.communicate()
		model = Word2Vec.load_word2vec_format(self.outputfile, binary=False)
		Logger.logr.info("Finished Loading WordDoc2Vec Model")
		return model 
		

	def getRepresentation(self, nx_G, initFromFile=False, initFile=""):
		"""
		Pipeline for representational learning for all nodes in a graph.
		"""
		n2vWalk= Node2VecWalk(nx_G, False, self.p, self.q)
		Logger.logr.info("Calculating Transition Probability")
		n2vWalk.preprocess_transition_probs()
		Logger.logr.info("Simulating Walks")
		walkInput = open("%s/node2vecwalk_%s.txt"%(self.DATADIR,str(initFromFile)))
		for walk in n2vWalk.simulate_walks(self.num_walks, self.walk_length):
			walkInput.write("%s%s"%walk) 
		Logger.logr.info("Learning Embeddings")
		return self.learnEmbeddings(walkInput, initFromFile, initFile)