#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod
from retrofitters.IterativeUpdateRetrofitter import IterativeUpdateRetrofitter
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
		self.p2vFile = os.environ["P2VECSENTRUNNEROUTFILE"]
		self.Graph = nx.Graph()
		self.sen2Vec = {}
		self.latReprName = "iterativeupdateunweighted"
		self.methodID = 5 
		
	
	def prepareData(self):
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
		retrofitter = IterativeUpdateRetrofitter(numIter=10, nx_Graph = self.Graph) 
		retrofitted_dict = retrofitter.retrofitWithIterUpdate(self.sen2Vec)
		Logger.logr.info("Retrofitted Dicitionary has %i objects" %len(retrofitted_dict))

	def generateSummary(self, gs):
		if gs <= 0: return 0
		itupdatevecFile = open("%s.p"%(self.retrofittedsen2vReprFile),"rb")
		itupdatevDict = pickle.load (itupdatevecFile)
		self.populateSummary(5, n2vDict)
		

	def runEvaluationTask(self):
		"""
		Generate Summary sentences for each document. 
		Write sentence id and corresponding metadata 
		into a file. 
		We should put isTrain=Maybe for the instances which 
		we do not want to incorporate in training and testing. 
		For example. validation set or unsup set
		"""
		itupdatevecFile = open("%s.p"%(self.retrofittedsen2vReprFile),"rb")
		itupdatevDict = pickle.load (itupdatevecFile)
		self.generateData(self.methodID, self.latReprName, n2vDict)
		self.runClassificationTask(self.methodID, self.latReprName)
		

	def doHouseKeeping(self):
		"""
		Here, we destroy the database connection.
		"""
		self.postgresConnection.disconnectDatabase()
