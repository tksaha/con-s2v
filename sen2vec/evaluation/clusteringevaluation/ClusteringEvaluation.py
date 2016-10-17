#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os 
import sys 
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from db_connector.PostgresPythonConnector import PostgresPythonConnector
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import math
import subprocess

class ClusteringEvaluation:
	"""
	Clustering Evaluation 
	"""
	__metaclass__ = ABCMeta

	def __writeClusteringData(self, result, fileToWrite, vecDict):
		"""
		"""
		for row_id in range(0, len(result)):
	 		id_ = result[row_id][0]
	 		topic = result[row_id][1]

	 		vec = vecDict[id_] 
	 		vec_str = ','.join(str(x) for x in vec)
	 		fileToWrite.write("%s,%s,%s%s"%(id_,vec_str,topic, os.linesep))

	def generateData(self, summaryMethodID, latReprName, vecDict):
		datafileToWrite = open("%s/%sclusterData_%i.csv"%(self.dataFolder,\
			 latReprName, summaryMethodID), "w")
		
		for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", summaryMethodID],\
			 	["sentence.topic","<>","'unsup'"] ], [], []):
				self.__writeClusteringData (result, datafileToWrite, vecDict)

	def runClusteringTask(self, summaryMethodID, latReprName):
		
