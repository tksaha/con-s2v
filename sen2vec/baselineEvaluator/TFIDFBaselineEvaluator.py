#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator
from baselineRunner.TFIDFBaselineRunner  import TFIDFBaselineRunner

class TFIDFBaselineEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		TFIDF baseline evaluator
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()
		
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		Logger.logr.info("Running TF-IDF Baseline")
		
		filePrefix = ""
		tfrunner = TFIDFBaselineRunner(self.dbstring)
		self.writeResults(pd, rbase, latent_space_size,\
			 tfrunner, filePrefix, f)
       