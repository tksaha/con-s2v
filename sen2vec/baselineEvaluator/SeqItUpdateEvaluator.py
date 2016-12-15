#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator
from baselineRunner.SeqItUpRetroRunner  import SeqItUpRetroRunner


class SeqItUpdateEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		Logger.logr.info ("Started Running Sequential Iterative Updated Iterator")

		filePrefix = "_unweighted"
		seqiterrunner = SeqItUpRetroRunner(self.dbstring)
		seqiterrunner.numIter = 20
		self.writeResults(pd, rbase, latent_space_size,\
			 seqiterrunner, filePrefix, f)