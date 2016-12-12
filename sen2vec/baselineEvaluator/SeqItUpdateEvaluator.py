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
from baselineRunner.SeqItUpRetroRunner  import SeqItUpRetroRunner

class SeqItUpdateEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		seqiterrunner = SeqItUpRetroRunner(self.dbstring)
		self.writeResults(pd, rbase, latent_space_size, seqiterrunner, "_unweighted", f)