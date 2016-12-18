#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineRunner.FastSentFHVersionRunner import FastSentFHVersionRunner
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner

class IterativeUpdatedRetrofitEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		Iterative Updated Retrofitter 
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)
		self.system_id_list = []

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()
		
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		Logger.logr.info("Starting running Iterative Updated Retrofitter")
		filePrefix = "_unweighted"
		iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
		iterrunner.nIter = 20
		self.system_id_list.append(iterrunner.system_id)
		self.writeResults(pd, rbase, latent_space_size,\
			 iterrunner, filePrefix, f) 