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

		# Hyperparameter 
		self.dictregBetaUNW = float(os.environ['REG_BETA_UNW'])
		self.dictregBetaW = float(os.environ['REG_BETA_W'])
		self.DictFile = os.environ['DICTREGDICT']


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

	def norm_word(self, word):
		stemmer = SnowballStemmer("english")
		return stemmer.stem(word.strip())

	def __write_neighbors (self, graph_list, file_to_write, weighted):
		
		max_neighbor = 0
		total_nodes = 0
		for graph in graph_list:
			max_nbr = self.__getMaxNeighbors(graph)
			if max_nbr > max_neighbor:
				max_neighbor = max_nbr
			total_nodes = total_nodes + graph.number_of_nodes()
		
		file_to_write.write("%s %s%s"%(total_nodes,max_neighbor, os.linesep))

		graph_id = 0
		for graph in graph_list:
			for nodes in graph.nodes():
				if graph_id ==0:
					file_to_write.write("%s "%label_sent(str(nodes)))
				else:
					file_to_write.write("%s "%str(nodes))
				nbrs = graph.neighbors(nodes)
				nbr_count = 0
				for nbr in nbrs:
					if weighted and graph_id==0:
						file_to_write.write("%s %s "%(label_sent(str(nbr)),graph[nodes][nbr]['weight']))
					else:
						if graph_id==0:
							file_to_write.write("%s %s "%(label_sent(str(nbr)),"1.0"))
						else:
							file_to_write.write("%s %s "%(str(nbr),"1.0"))
					nbr_count = nbr_count +1 

				if nbr_count < max_neighbor:
					for  x in range(nbr_count, max_neighbor):
						file_to_write.write("%s %s " %("-1","0.0"))

				file_to_write.write("%s"%os.linesep)

			graph_id = graph_id + 1

		file_to_write.flush()
		file_to_write.close()


	def prepareData(self, pd):
		if pd <= 0: return 0
		self.Graph = nx.read_gpickle(self.graphFile)

		graph_list = []
		graph_list.append(self.Graph)

		fileNames = []
		for root, directories, files in os.walk(self.dictDir):
			for file in files:
				if 	file == self.DictFile:
					fileName = os.path.join(root, file)
					fileNames.append(fileName)

		for file in fileNames:
			Logger.logr.info("Working for dictionary file %s"%file)
			nx_G = nx.Graph()
			for line in open(file):
				words = line.lower().strip().split()
				if len(words) > 1:
					first_word = self.norm_word(words[0])
					for words in words[1:]:
						nx_G.add_edge(first_word, self.norm_word(words))
			graph_list.append(nx_G)

		neighbor_file_w = open("%s_neighbor_w.txt"%(self.dictRegSen2vReprFile), "w")
		neighbor_file_unw = open("%s_neighbor_unw.txt"%(self.dictRegSen2vReprFile), "w")

		self.__write_neighbors (graph_list, neighbor_file_w, weighted=True)
		self.__write_neighbors (graph_list, neighbor_file_unw, weighted=False)
		self.Graph = nx.Graph()

	def __dumpVecs(self, reprFile, vecFile, vecRawFile):

		vModel = Word2Vec.load_word2vec_format(reprFile, binary=False)
		
		vec_dict = {}
		vec_dict_raw = {}

		for nodes in self.Graph.nodes():
			vec = vModel[label_sent(str(nodes))]
			vec_dict_raw[int(nodes)] = vec 
			vec_dict[int(nodes)] = vec /  ( np.linalg.norm(vec) +  1e-6)

		pickle.dump(vec_dict, vecFile)
		pickle.dump(vec_dict_raw, vecRawFile)


	def runTheBaseline(self, rbase, latent_space_size):
		if rbase <=0: return 0

		self.Graph = nx.read_gpickle(self.graphFile)

		wordDoc2Vec = WordDoc2Vec()
		wPDict = wordDoc2Vec.buildWordDoc2VecParamDict()
		wPDict["cbow"] = str(0) 
		wPDict["sentence-vectors"] = str(1)
		wPDict["min-count"] = str(0)
		wPDict["train"] = "%s.txt"%self.sentsFile
		
		wPDict["size"]= str(latent_space_size * 2)
		args = []

######################### Working for Weighted Neighbor File ##################	
		neighborFile = 	"%s_neighbor_w.txt"%(self.dictRegSen2vReprFile)
		wPDict["output"] = "%s_neighbor_w"%(self.dictRegSen2vReprFile)
		wPDict["neighborFile"], wPDict["beta"] = neighborFile, str(self.dictregBetaW)
		args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 2)
		self._runProcess (args)
		self.__dumpVecs(wPDict["output"],\
			 open("%s.p"%wPDict["output"], "wb"),\
			 open("%s_raw.p"%wPDict["output"], "wb"))

		
######################### Working for UnWeighted Neighbor File ###################		
		neighborFile = 	"%s_neighbor_unw.txt"%(self.dictRegSen2vReprFile)
		wPDict["output"] = "%s_neighbor_unw"%(self.dictRegSen2vReprFile)
		wPDict["neighborFile"], wPDict["beta"] = neighborFile, str(self.dictregBetaUNW)
		args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 2)
		self._runProcess (args)
		self.__dumpVecs(wPDict["output"],\
				open("%s.p"%wPDict["output"], "wb"),\
				open("%s_raw.p"%wPDict["output"], "wb"))
		self.Graph = nx.Graph()


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
		
	def __runEval(self, summaryMethodID, vecFileName, reprName):
		vecFile = open("%s.p"%vecFileName,"rb")
		vDict = pickle.load (vecFile)
		self._runClassification(summaryMethodID, reprName, vDict)

		vecFile = open("%s_raw.p"%vecFileName, "rb")
		vDict = pickle.load (vecFile)
		self._runClassification(summaryMethodID, "%s_raw"%reprName, vDict)
		self._runClustering(summaryMethodID, "%s_raw"%reprName, vDict)

	def runEvaluationTask(self):
		summaryMethodID = 2 
		Logger.logr.info("Starting Dict Regularized Sentence 2 Vector Evaluation")
		
		regvecFile = "%s_neighbor_w"%(self.dictRegSen2vReprFile)
		reprName = "%s_neighbor_w"%self.latReprName
		self.__runEval(summaryMethodID, regvecFile, reprName)

		regvecFile = "%s_neighbor_unw"%(self.dictRegSen2vReprFile)
		reprName = "%s_neighbor_unw"%self.latReprName
		self.__runEval(summaryMethodID, regvecFile, reprName)
		
	def doHouseKeeping(self):
		self.postgresConnection.disconnectDatabase()