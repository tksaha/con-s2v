#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator
from baselineRunner.SkipThoughtRunner import SkipThoughtRunner


class SkipThoughtEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		Word Vector Averaging Evaluator
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)
		self.filePrefix = ""
		self.system_id_list = []

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		f.write("[Skip-Thought Baseline] (No Tuning) %s" %(os.linesep))	
		sthoughtBaseline = SkipThoughtRunner(self.dbstring)
		self.system_id_list.append(sthoughtBaseline.system_id)
		self.writeResults(pd, rbase, latent_space_size, sthoughtBaseline, self.filePrefix, f)