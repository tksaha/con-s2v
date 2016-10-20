#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math
from abc import ABCMeta, abstractmethod
import networkx as nx 
import pandas as pd
from log_manager.log_config import Logger
from sklearn.dummy import DummyClassifier
from sklearn import linear_model
import operator
import numpy as np 
import subprocess 
from db_connector.PostgresPythonConnector import PostgresPythonConnector
from evaluation.classificationevaluaiton.ClassificationEvaluation import ClassificationEvaluation 
from evaluation.clusteringevaluation.ClusteringEvaluation import ClusteringEvaluation 

class BaselineRunner:
	def __init__(self, dbstring, **kwargs):
		"""
		"""
		self.postgresConnection = PostgresPythonConnector(dbstring)

	def _runProcess (self,args): 
		Logger.logr.info(args)
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = proc.communicate()
		if 	proc.returncode != 0: 
			Logger.logr.error("Process haven't terminated successfully")
			Logger.logr.info(out)
			Logger.logr.info(err)
			sys.exit(1)

	def _runClassificationValidation(self, summaryMethodID,  reprName, vDict):
		classeval = ClassificationEvaluation(postgres_connection=self.postgresConnection)
		classeval.generateDataValidation(summaryMethodID, reprName, vDict)
		classeval.runClassificationTask(summaryMethodID, reprName)

	def _runClusteringValidation(self, summaryMethodID, reprName, vDict):
		clusterEval = ClusteringEvaluation(postgres_connection=self.postgresConnection)
		clusterEval.generateDataValidation(summaryMethodID, reprName, vDict)
		clusterEval.runClusteringTask(summaryMethodID, reprName)

	def _runClassification(self, summaryMethodID,  reprName, vDict):
		classeval = ClassificationEvaluation(postgres_connection=self.postgresConnection)
		classeval.generateData(summaryMethodID, reprName, vDict)
		classeval.runClassificationTask(summaryMethodID, reprName)

	def _runClustering(self, summaryMethodID, reprName, vDict):
		clusterEval = ClusteringEvaluation(postgres_connection=self.postgresConnection)
		clusterEval.generateData(summaryMethodID, reprName, vDict)
		clusterEval.runClusteringTask(summaryMethodID, reprName)

	@abstractmethod
	def prepareData(self):
		"""
		"""
		pass

	@abstractmethod
	def runTheBaseline(self):
		"""
		"""
		pass

	@abstractmethod
	def runEvaluationTask(self):
		"""
		"""
		pass

	@abstractmethod
	def generateSummary(self):
		"""
		"""
		pass 
	@abstractmethod
	def doHouseKeeping(self):
		"""
		This method will close existing database connections and 
		other resouces it has used. 
		"""
		pass