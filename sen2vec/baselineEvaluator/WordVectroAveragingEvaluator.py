#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineRunner.WordVectorAveragingRunner import WordVectorAveragingRunner


class WordVectorAveragingEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		Word Vector Averaging Evaluator
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()
		
		for window in self.window_size_list:
			Logger.logr.info("[WVAvg Baseline] Starting Running  for Window = %s" %window)				
				wvBaseline = WordVectorAveragingRunner (self.dbstring)
				if 	window == self.window_size_list[0]: 
					wvBaseline.prepareData(pd)		
				wvBaseline.window_size = window
			self.metric[beta] = self.evaluate(wvBaseline, filePrefix,\
					 latent_space_size)	
			Logger.logr.info("[WVAvg Baseline] %s for window %s = %s"\
			 %(self.metric_str, window, self.metric[beta]))
			
		window_opt_avg = max(self.metric, key=self.metric.get)
		Logger.logr.info("[WVAvg Baseline] Optimal window=%s" %(window_opt_avg))	
		optPDict['window-opt-avg'] = window_opt_avg
		f.write("[WVAvg Baseline] Optimal Window : %.2f%s" %(unw_opt_avg, os.linesep))
		f.write("[WVAvg Baseline] %ss: %s%s" %(self.metric_str, self.metric, os.linesep))
		f.flush()
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		filePrefix = ""
		f.write("[WVAvg Baseline] Optimal Window  is: %s%s" %(optPDict["window-opt-avg"], os.linesep))	
		wvBaseline = WordVectorAveragingRunner(self.dbstring)
		wvBaseline.window_size = optPDict["window-opt-avg"]
		self.writeResults(pd, rbase, latent_space_size, wvBaseline, filePrefix, f)