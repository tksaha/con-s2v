#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator


class Node2VecEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		Joint Learning Sen2Vec evaluator
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()
		
		n2vBaseline = Node2VecRunner(self.dbstring)
		n2vBaseline.prepareData(pd)
		n2vBaseline.runTheBaseline(rbase, latent_space_size, True)
		
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		# We don't use node2vec as a baseline for jointlearning work
		pass