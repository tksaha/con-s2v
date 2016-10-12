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
from baselineRunner.BaselineRunner import BaselineRunner





class BaselineRunner:
	def __init__(self, dbstring, **kwargs):
		"""
		"""
		pass

	def _runProcess (args): 
		Logger.logr.info(args)
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = proc.communicate()
		if 	proc.returncode != 0: 
			Logger.logr.error("Process haven't terminated successfully")
			sys.exit(1)


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