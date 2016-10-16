#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 
from gensim.models import Word2Vec
from log_manager.log_config import Logger 
from baselineRunner.BaselineRunner import BaselineRunner
import pickle
import math 
import operator 
import multiprocessing 
import subprocess 
import numpy as np 
from word2vec.WordDoc2Vec import WordDoc2Vec
from nltk.stem.snowball import SnowballStemmer
from summaryGenerator.SummaryGenerator import SummaryGenerator


label_sent = lambda id_: 'SENT_%s' %(id_)


class DictRegularizedSen2VecRunner(BaselineRunner): 

	def __init__(self, *args, **kwargs):
		BaselineRunner.__init__(self, *args, **kwargs)
		self.dictRegSen2vReprFile = os.environ["DICTREGSEN2VECREPRFILE"]
		self.dataDir = os.environ['TRTESTFOLDER']
		self.dictDir = os.environ['DICTDIR']
		self.sentsFile = os.environ['P2VCEXECSENTFILE']
		self.Graph = nx.Graph()
		self.cores = multiprocessing.cpu_count()
		self.graphFile = os.environ["GRAPHFILE"]
		self.latReprName = "dictreg_s2v"
		self.postgresConnection.connectDatabase()

	def __getMaxNeighbors(self, nx_G):
		"""
		Calculates the maximum number of neighbors.
		"""
		max_neighbor = 0 
		for nodes in nx_G.nodes():
			nbrs = nx_G.neighbors(nodes)
			if len(nbrs) > max_neighbor:
				max_neighbor = len(nbrs)

		return max_neighbor


	def prepareData(self, pd):
		if pd <= 0: return 0
		self.Graph = nx.read_gpickle(self.graphFile)

		fileNames = []
		for root, directories, files in os.walk(self.dictDir):
			for file in files:
				fileName = os.path.join(root, file)
				fileNames.append(fileName)


		dict_graphs = []
		for file in fileNames:
			nx_G = nx.Graph()
			for line in open(file):
				words = line.lower().strip().split()
				if len(words) > 1:
					first_word = norm_word(words[0])
					for words in words[1:]:
						nx_G.add_edge(first_word, norm_word(words))
			dict_graphs.append(nx_G)




	def runTheBaseline(self, rbase, latent_space_size):
		if rbase <=0: return 0


	def generateSummary(self, gs, methodId, filePrefix,\
		 lambda_val=1.0, diversity=False):
		if gs <= 0: return 0
		dictregsentvecFile = open("%s%s.p"%(self.dictRegSen2vReprFile,\
			 filePrefix),"rb")
		dictregsentvDict = pickle.load (dictregsentvecFile)
		
		summGen = SummaryGenerator (diverse_summ=diversity,\
			 postgres_connection = self.postgresConnection,\
			 lambda_val = lambda_val)

		summGen.populateSummary(methodId, dictregsentvDict)
		

	def runEvaluationTask(self):
		pass

	def doHouseKeeping(self):
		pass