#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator


class FastSentVariantEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		Fast Sent Variant Runner
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()
		
		filePrefix = ""
		for lambda_ in self.lambda_list:
			Logger.logr.info("[CON-S2V-S Baseline] Starting Running for lambda = %s" %lambda_)				
			os.environ["FULL_DATA"]=str(0)
			os.environ["LAMBDA"]=str(lambda_)
			fsent =  FastSentVariantRunner(self.dbstring)	
			fsent.window = optPDict["window"]
			if  lambda_ == lambda_list[0]:
				fsent.prepareData(1)
			self.metric[lambda_] = self.evaluate(fsent, filePrefix, latent_space_size)
			Logger.logr.info("[CON-S2V-S Baseline] %s for lambda %.2f = %s"\
			 	%(self.metric_str, lambda_, self.metric[lambda_]))
			
		fsent_lambda = max(self.metric, key=self.metric.get)
		optPDict['con-s2v-s-lambda'] = fsent_lambda
		f.write("[CON-S2V-S Baseline] Optimal lambda : %.2f%s" %(fsent_lambda, os.linesep))
		f.write("[CON-S2V-S Baseline] %ss: %s%s" %(self.metric_str, self.metric, os.linesep))
		f.flush()
			
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):

		os.environ["FULL_DATA"] = str(0)
		os.environ["LAMBDA"] = optPDict['con-s2v-s-lambda']	
		fsent =  FastSentVariantRunner(self.dbstring)	
		fsent.window = optPDict["window"]
		self.writeResults(pd, rbase, latent_space_size, fsent, filePrefix, f)
