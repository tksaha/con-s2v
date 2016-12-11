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
from baselineRunner.SequentialRegularizedSen2VecRunner import SequentialRegularizedSen2VecRunner

class SeqRegSentEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		metric = {}
		metric_str = "f1"
		if os.environ['VALID_FOR'] == 'CLUST':
		   metric_str = "AdjMIScore"

		seqregs2v = SequentialRegularizedSen2VecRunner(self.dbstring)
		for beta in self.beta_list:
			Logger.logr.info("Starting Running Seq Regularized "+\
				" Baseline for Beta = %s" %beta)
			seqregs2v.seqregunw_beta = beta
			seqregs2v.window_size = optPDict["window"]
			if beta==0.3:
			   seqregs2v.prepareData(1)
			seqregs2v.runTheBaseline(1, latent_space_size)
			seqregs2v.runEvaluationTask()
			metric[beta] = self._getF1("%s_neighbor_unw"%seqregs2v.latReprName)	
			Logger.logr.info("UNW_%s for %s = %s" %(metric_str, beta, metric[beta]))
			
		unw_opt_seq_reg = max(metric, key=metric.get)
		Logger.logr.info("Optimal seqregBetaUNW=%s" %(unw_opt_seq_reg))
			
		optPDict['unw_opt_seq_reg'] = unw_opt_seq_reg
		f.write("Optimal REG BetaUNW : %.2f%s" %(unw_opt_seq_reg, os.linesep))
		f.write("Seq REG BetaUNW %ss: %s%s" %(metric_str, metric, os.linesep))
		f.flush()
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		
		f.write("Optimal Window for seq reg s2v is: %s%s" %(optPDict["unw_opt_seq_reg"], os.linesep))	
		seqregs2v = SequentialRegularizedSen2VecRunner(self.dbstring)
		seqregs2v.seqregunw_beta = optPDict['unw_opt_seq_reg']
		seqregs2v.window_size = optPDict["window"]
		seqregs2v.prepareData(pd)		
		seqregs2v.runTheBaseline(rbase,latent_space_size)
		seqregs2v.runEvaluationTask()
		self._writeResult("%s_neighbor_unw"%seqregs2v.latReprName, f)
		seqregs2v.doHouseKeeping()	
		f.flush()


		