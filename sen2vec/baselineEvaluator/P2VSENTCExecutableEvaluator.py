#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator
from baselineRunner.P2VSENTCExecutableRunner  import P2VSENTCExecutableRunner



class P2VSENTCExecutableEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		DM+DBOW baseline runner
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)
		self.filePrefix = ""

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()

		for window in self.window_size_list:
			Logger.logr.info("[S2V Baseline] Starting Running for Window = %s" %window)				
			paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			paraBaseline.window_size = window
			if 	window == self.window_size_list[0]:  
				self.postgres_recorder.truncateSummaryTable()
				paraBaseline.prepareData(1)		
				paraBaseline.runTheBaseline(1,latent_space_size)
			if window == self.window_size_list[0]:
				paraBaseline.generateSummary(1)
			self.metric[window] = self.evaluate(paraBaseline, self.filePrefix, latent_space_size)
			Logger.logr.info("[S2V Baseline] %s for window %s = %s"\
			 	%(self.metric_str, window, self.metric[window]))
			
		window_opt = max(self.metric, key=self.metric.get)
		Logger.logr.info("[S2V Baseline] Optimal window=%s" %(window_opt))	
		optPDict['window'] = window_opt
		f.write("[S2V Baseline] Optimal Window : %s%s" %(optPDict['window'], os.linesep))
		f.write("[S2V Baseline] %ss: %s%s" %(self.metric_str, self.metric, os.linesep))
		f.flush()

		paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
		self.postgres_recorder.truncateSummaryTable()
		paraBaseline.window_size = optPDict['window']
		paraBaseline.prepareData(1)		
		paraBaseline.runTheBaseline(1,latent_space_size)
		paraBaseline.generateSummary(1)

		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		
		f.write("[S2V Baseline] Optimal Window  is: %s%s" %(optPDict["window"], os.linesep))	
		paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
		paraBaseline.window_size = optPDict["window"]
		self.writeResults(pd, rbase, latent_space_size, paraBaseline, self.filePrefix, f)