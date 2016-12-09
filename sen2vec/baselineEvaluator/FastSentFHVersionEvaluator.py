#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
from abc import ABCMeta, abstractmethod
from baselineRunner.FastSentFHVersionRunner import FastSentFHVersionRunner
from baselinevaluator.BaselineEvaluator import BaselineEvaluator


class FastSentFHVersionEvalutor(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		metric = {}
		metric_str = "f1"
		if os.environ['VALID_FOR'] == 'CLUST':
		   metric_str = "AdjMIScore"

		window_opt = None 
		for window in self.window_size_list:
			Logger.logr.info("Starting Running FastsentFHVersion "+\
					" Baseline for Window = %s" %window)				
			fhBaseline =  FastSentFHVersionRunner(self.dbstring, autoencode=False)
			fhBaseline.window  = window
			if 	window == self.window_size_list[0]: 
				fhBaseline.prepareData(1)		
			metric[window] = evaluate(fhBaseline)
			Logger.logr.info("%s for window %s = %s" %(metric_str, window, metric[window]))
		window_opt_avg = max(metric, key=metric.get) 
		f.write("Optimal window size for %s in fhbaseline is %s%s"\
				%(metric_str, window_opt, os.linesep))
		f.write("%ss: %s%s" %(metric_str, metric, os.linesep))
		f.flush()
		optPDict["fh-window-avg"] = window_opt_avg


		metric = {}
		window_opt = None 
		for window in self.window_size_list:
			Logger.logr.info("Starting Running FastsentFHVersion (AE) "+\
					" Baseline for Window = %s" %window)				
			fhBaseline =  FastSentFHVersionRunner(self.dbstring, autoencode=True)
			fhBaseline.window  = window
			if 	window == self.window_size_list[0]: 
				fhBaseline.prepareData(1)		
			metric[window] = evaluate(fhBaseline)
			Logger.logr.info("%s for window %s = %s" %(metric_str, window, metric[window]))
		window_opt_avg = max(metric, key=metric.get) 
		f.write("Optimal window size for %s in fhbaseline is %s%s"\
				%(metric_str, window_opt, os.linesep))
		f.write("%ss: %s%s" %(metric_str, metric, os.linesep))
		f.flush()

		optPDict["fh-ae-window-avg"] = window_opt_avg
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		f.write("Optimal Window for fh is: %s%s" %(optPDict["fh-window-avg"], os.linesep))	
		fhBaseline =  FastSentFHVersionRunner(self.dbstring, autoencode=False)
		fhbaseline.window = optPDict["fh-window-avg"]
		self.writeResults(pd, rbase, latent_space_size, fhBaseline, f)

		f.write("Optimal Window for fh (+AE) is: %s%s" %(optPDict["fh-ae-window-avg"], os.linesep))	
		fhBaseline =  FastSentFHVersionRunner(self.dbstring, autoencode=True)
		fhBaseline.window = optPDict["fh-ae-window-avg"]
		self.writeResults(pd, rbase, latent_space_size, fhBaseline, f)




		