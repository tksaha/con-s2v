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


class SeqItUpdateEvalutor(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		metric = {}
		metric_str = "f1"
		if os.environ['VALID_FOR'] == 'CLUST':
		   metric_str = "AdjMIScore"

		return optPDict
		
	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		pass
		