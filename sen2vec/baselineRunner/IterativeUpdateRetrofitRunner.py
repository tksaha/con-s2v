#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod
from retrofitters.IterativeUpdateRetrofitter import IterativeUpdateRetrofitter
from retrofitters.WeightedIterativeUpdateRetrofitter import WeightedIterativeUpdateRetrofitter
from retrofitters.RandomWalkIterativeUpdateRetrofitter import RandomWalkIterativeUpdateRetrofitter
from baselineRunner.BaselineRunner import BaselineRunner
import networkx as nx 
import pickle 
import numpy as np 
from log_manager.log_config import Logger 



class IterativeUpdateRetrofitRunner(BaselineRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)
		self.retrofittedsen2vReprFile = os.environ["ITERUPDATESEN2VECFILE"]
		self.graphFile = os.environ["GRAPHFILE"]
		self.p2vFile = os.environ['P2VCEXECOUTFILE']
		self.Graph = nx.Graph()
		self.postgresConnection.connectDatabase()
		self.sen2Vec = {}
		self.latReprName = "iterativeupdate"
		self.methodID = 5 
		
	
	def prepareData(self, pd):
		"""
		"""
		pass 

	def runTheBaseline(self, rbase):
		"""
		Write down the Iterative update vector
		"""
		self.Graph = nx.read_gpickle(self.graphFile)
		p2vfileToRead = open ("%s.p" %self.p2vFile, "rb")
		self.sen2Vec = pickle.load(p2vfileToRead)
		Logger.logr.info("Dictionary has %i objects" % len(self.sen2Vec))


		retrofitter = IterativeUpdateRetrofitter(numIter=20, nx_Graph = self.Graph) 
		retrofitted_dict = retrofitter.retrofitWithIterUpdate(self.sen2Vec)
		Logger.logr.info("Retrofitted Dicitionary has %i objects" %len(retrofitted_dict))

		iterupdatevecFile = open("%s_unweighted.p"%(self.retrofittedsen2vReprFile),"wb")
		pickle.dump(retrofitted_dict,iterupdatevecFile )

		wretrofitter = WeightedIterativeUpdateRetrofitter(numIter=20, nx_Graph = self.Graph)
		retrofitted_dict = wretrofitter.retrofitWithIterUpdate(self.sen2Vec, alpha =-1)
		iterupdatevecFile = open("%s_weighted.p"%(self.retrofittedsen2vReprFile),"wb")
		pickle.dump(retrofitted_dict, iterupdatevecFile)

		randomwalkretrofitter = RandomWalkIterativeUpdateRetrofitter(numIter=10)
		rand_retrofitted_dict = randomwalkretrofitter.retrofitWithIterUpdate(self.sen2Vec)
		rand_iterupdateFile = open("%s_randomwalk.p"%(self.retrofittedsen2vReprFile),"wb")
		pickle.dump(rand_retrofitted_dict, rand_iterupdateFile)


	def generateSummary(self, gs, methodId, filePrefix):
		if gs <= 0: return 0
		itupdatevecFile = open("%s%s.p"%(self.retrofittedsen2vReprFile, filePrefix),"rb")
		itupdatevDict = pickle.load (itupdatevecFile)
		self.populateSummary(methodId, itupdatevDict)
		

	def runEvaluationTask(self):
		"""
		Generate Summary sentences for each document. 
		Write sentence id and corresponding metadata 
		into a file. 
		We should put isTrain=Maybe for the instances which 
		we do not want to incorporate in training and testing. 
		For example. validation set or unsup set
		"""
		itupdatevecFile = open("%s_unweighted.p"%(self.retrofittedsen2vReprFile),"rb")
		itupdatevDict = pickle.load (itupdatevecFile)
		reprName = "%s_unweighted"%self.latReprName
		self.generateData(6, reprName, itupdatevDict)
		self.runClassificationTask(6, reprName)
		
		itupdatevecFile = open("%s_weighted.p"%(self.retrofittedsen2vReprFile),"rb")
		itupdatevDict = pickle.load (itupdatevecFile)
		reprName = "%s_weighted"%self.latReprName
		self.generateData(7, reprName, itupdatevDict)
		self.runClassificationTask(7, reprName)

		rand_itervecFile = open("%s_randomwalk.p"%(self.retrofittedsen2vReprFile),"rb")
		rand_itupdatevDict = pickle.load(rand_itervecFile) 
		reprName = "%s_randomwalk"%self.latReprName
		self.generateData(2, reprName, rand_itupdatevDict)
		self.runClassificationTask(2, reprName)

	def doHouseKeeping(self):
		"""
		Here, we destroy the database connection.
		"""
		self.postgresConnection.disconnectDatabase()
