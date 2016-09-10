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
		
	
	def prepareData(self):
		"""
		"""
		self.Graph = nx.read_gpickle(self.graphFile)
		p2vfileToRead = open ("%s.p" %self.p2vFile, "rb")
		while True: 
			try:
				sent_dict = pickle.load(p2vfileToRead)
				id_ = sent_dict["id"]
				vec = sent_dict["vec"]
				vec = vec / np.linalg.norm(vec)
				self.sen2Vec[id_] = vec 
			except Exception as e:
				Logger.logr.info(str(e))
				break 

		Logger.logr.info("Dictionary has %i objects" % len(self.sen2Vec))


	def runTheBaseline(self):
		"""
		"""
		retrofitter = IterativeUpdateRetrofitter(numIter=100, nx_Graph = self.Graph) 
		retrofitted_dict = retrofitter.retrofitWithIterUpdate(self.sen2Vec)
		Logger.logr.info("Retrofitted Dicitionary has %i objects" %len(retrofitted_dict))

	def runEvaluationTask(self):
		"""
		"""
		pass

	
	def prepareStatisticsAndWrite(self):
		"""
		"""
		pass

