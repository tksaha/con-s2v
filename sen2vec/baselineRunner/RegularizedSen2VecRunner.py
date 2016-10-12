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



label_sent = lambda id_: 'SENT_%s' %(id_)


class RegularizedSen2VecRunner(BaselineRunner): 

	def __init__(self, *args, **kwargs):
		BaselineRunner.__init__(self, *args, **kwargs)
		self.regsen2vReprFile = os.environ["REGSEN2VECREPRFILE"]
		self.dataDir = os.environ['TRTESTFOLDER']
		self.sentsFile = os.environ['P2VCEXECSENTFILE']
		self.Graph = nx.Graph()
		self.cores = multiprocessing.cpu_count()
		self.graphFile = os.environ["GRAPHFILE"]
		self.latReprName = "reg_s2v"
		self.postgresConnection.connectDatabase()
	
	def __getMaxNeighbors(self):
		"""
		Calculates the maximum number of neighbors.
		"""
		max_neighbor = 0 
		for nodes in self.Graph.nodes():
			nbrs = self.Graph.neighbors(nodes)
			if len(nbrs) > max_neighbor:
				max_neighbor = len(nbrs)

		return max_neighbor

	def __write_neighbors (self, max_neighbor, file_to_write, weighted):
		file_to_write.write("%s %s%s"%(self.Graph.number_of_nodes(),max_neighbor, os.linesep))

		for nodes in self.Graph.nodes():
			file_to_write.write("%s "%label_sent(str(nodes)))
			nbrs = self.Graph.neighbors(nodes)
			nbr_count = 0
			for nbr in nbrs:
				if weighted:
					file_to_write.write("%s %s "%(label_sent(str(nbr)),self.Graph[nodes][nbr]['weight']))
				else:
					file_to_write.write("%s %s "%(label_sent(str(nbr)),"1.0"))
				nbr_count = nbr_count +1 

			if nbr_count < max_neighbor:
				for  x in range(nbr_count, max_neighbor):
					file_to_write.write("%s %s " %("-1","0.0"))

			file_to_write.write("%s"%os.linesep)

		file_to_write.flush()
		file_to_write.close()

	def prepareData(self, pd):
		"""
		It prepares neighbor data for regularized sen2vec. 
		The first line of the file will indicate how nodes 
		are in the file and max number of neighbors. If a 
		particular node has less number of neighbors than the 
		maximum numbers then "-1" should be written as 
		neighbor. For the unweighted version, all weights should 
		be 1.0. 
		"""
		if pd <= 0: return 0 
		self.Graph = nx.read_gpickle(self.graphFile)
		max_neighbor = self.__getMaxNeighbors()

		neighbor_file_w = open("%s_neighbor_w.txt"%(self.regsen2vReprFile), "w")
		neighbor_file_unw = open("%s_neighbor_unw.txt"%(self.regsen2vReprFile), "w")

		self.__write_neighbors (max_neighbor, neighbor_file_w, weighted=True)
		self.__write_neighbors (max_neighbor, neighbor_file_unw, weighted=False)


	def __dumpVecs(self, reprFile, vecFile):

		vModel = Word2Vec.load_word2vec_format(reprFile, binary=False)
			
		vec_dict = {}
		for nodes in self.Graph.nodes():
			print  (label_sent(str(nodes)))
			vec = vModel[label_sent(str(nodes))]
			vec_dict[int(nodes)] = vec /  ( np.linalg.norm(vec) +  1e-6)

		pickle.dump(vec_dict, vecFile)

	def __printLogs (self, out, err):
		Logger.logr.info(out)
		Logger.logr.info(err) 

	def runTheBaseline(self, rbase, latent_space_size):
		if rbase <= 0: return 0 

		wordDoc2Vec = WordDoc2Vec()
		wPDict = wordDoc2Vec.buildWordDoc2VecParamDict()
		wPDict["cbow"] = str(0) 
		wPDict["sentence-vectors"] = str(1)
		wPDict["min-count"] = str(0)
		wPDict["train"] = "%s.txt"%self.sentsFile
		
		wPDict["size"]= str(latent_space_size * 2)
		args = []

######################### Working for Weighted Neighbor File ##################	
		neighborFile = 	"%s_neighbor_w.txt"%(self.regsen2vReprFile)
		wPDict["output"] = "%s_neighbor_w"%(self.regsen2vReprFile)
		wPDict["neighborFile"], wPDict["reg-nbr"] = neighborFile, str(1)
		args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 2)
		Logger.logr.info(args)
		process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = process.communicate()
		self.__printLogs(out, err)
		self.__dumpVecs(wPDict["output"], open("%s.p"%wPDict["output"], "wb"))

		
######################### Working for UnWeighted Neighbor File ###################
		
		neighborFile = 	"%s_neighbor_unw.txt"%(self.regsen2vReprFile)
		wPDict["output"] = "%s_neighbor_unw"%(self.regsen2vReprFile)
		wPDict["neighborFile"], wPDict["reg-nbr"] = neighborFile, str(1)
		args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 2)
		Logger.logr.info(args)
		process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = process.communicate()
		self.__printLogs(out, err)
		self.__dumpVecs(wPDict["output"], open("%s.p"%wPDict["output"], "wb"))


	def generateSummary(self, gs, methodId, filePrefix):
		if gs <= 0: return 0
		regsentvecFile = open("%s%s.p"%(self.regsen2vReprFile, filePrefix),"rb")
		regsentvDict = pickle.load (regsentvecFile)
		self.populateSummary(methodId, itupdatevDict)
		

	def runEvaluationTask(self):
		summaryMethodID = 2 
		self.Graph = nx.read_gpickle(self.graphFile)

		self.__dumpVecs("%s_neighbor_w"%(self.regsen2vReprFile), open("%s_neighbor_w.p"%(self.regsen2vReprFile), "wb"))
		regvecFile = open("%s_neighbor_w.p"%(self.regsen2vReprFile),"rb")
		regvDict = pickle.load (regvecFile)
		reprName = "%s_neighbor_w"%self.latReprName
		self.generateData(summaryMethodID, reprName, regvDict)
		self.runClassificationTask(summaryMethodID, reprName)
		

		# regvecFile = open("%s_neighbor_unw.p"%(self.regsen2vReprFile),"rb")
		# regvDict = pickle.load (regvecFile)
		# reprName = "%s_neighbor_unw"%self.latReprName
		# self.generateData(summaryMethodID, reprName, regvDict)
		# self.runClassificationTask(summaryMethodID, reprName)
		
		
	def doHouseKeeping(self):
		"""
		Here, we destroy the database connection.
		"""
		self.postgresConnection.disconnectDatabase()
	
	