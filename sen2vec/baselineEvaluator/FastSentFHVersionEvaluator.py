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


class FastSentFHVersionEvalutor(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()
		window_opt_avg = None 
		fhBaseline =  FastSentFHVersionRunner(self.dbstring, autoencode=True)
		for window in self.window_size_list:
			Logger.logr.info("Starting Running FastsentFHVersion (AE) "+\
					" Baseline for Window = %s" %window)				
			fhBaseline.window  = window
			if 	window == self.window_size_list[0]: 
				fhBaseline.prepareData(1)		
			self.metric[window] = self.evaluate(fhBaseline, "",latent_space_size)
			Logger.logr.info("%s for window %s = %s" %(self.metric_str, window, self.metric[window]))
		window_opt_avg = max(self.metric, key=self.metric.get) 
		f.write("Optimal window size for %s in fhbaseline is %s%s"\
				%(self.metric_str, window_opt_avg, os.linesep))
		f.write("%ss: %s%s" %(self.metric_str, self.metric, os.linesep))
		f.flush()

		optPDict["fh-ae-window-avg"] = window_opt_avg
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		f.write("Optimal Window for fh (+AE) is: %s%s" %(optPDict["fh-ae-window-avg"], os.linesep))	
		fhBaseline =  FastSentFHVersionRunner(self.dbstring, autoencode=True)
		fhBaseline.window = optPDict["fh-ae-window-avg"]
		self.writeResults(pd, rbase, latent_space_size, fhBaseline, "", f)




		