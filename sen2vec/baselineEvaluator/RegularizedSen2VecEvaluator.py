#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator


class RegularizedSen2VecEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		TFIDF baseline evaluator
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()
		
		filePrefix = "_neighbor_unw"
		unw_opt_reg = None 

		regs2v = RegularizedSen2VecRunner(self.dbstring)
		for beta in self.beta_list:
			Logger.logr.info("[RegS2V] Starting Running "+\
				" Baseline for Beta = %s" %beta)
			regs2v.regunw_beta = beta
			regs2v.window_size = optPDict["window"]
			if beta== self.beta_list[0]:
			   regs2v.prepareData(1)

			self.metric[beta] = self.evaluate(regs2v,\
			 filePrefix, latent_space_size)	
			Logger.logr.info("[RegS2V] UNW_%s for %s = %s"\
				%(self.metric_str, beta, self.metric[beta]))
			
		unw_opt_reg = max(self.metric, key=self.metric.get)
		Logger.logr.info("[RegS2V] Optimal BetaUNW=%s" %(unw_opt_reg))	
		optPDict['unw-opt-reg'] = unw_opt_reg
		f.write("[RegS2V] BetaUNW : %.2f%s" %(unw_opt_reg, os.linesep))
		f.write("[RegS2V] BetaUNW %ss: %s%s" %(self.metric_str,\
			 self.metric, os.linesep))
		f.flush()
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		filePrefix = "_neighbor_unw"
		f.write("[RegS2V] Optimal beta is: %s%s" %(optPDict["unw-opt-reg"], os.linesep))	
		regs2v = RegularizedSen2VecRunner(self.dbstring)
		regs2v.seqregunw_beta = optPDict['unw-opt-reg']
		regs2v.window_size = optPDict["window"]
		self.writeResults(pd, rbase, latent_space_size,\
			 regs2v, filePrefix, f)
		