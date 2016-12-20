#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator
from baselineRunner.JointLearningSen2VecRunner import  JointLearningSen2VecRunner

class JointLearningSen2VecEvaluator(BaselineEvaluator):
	def __init__(self, *args, **kwargs):
		"""
		Joint Learning Sen2Vec evaluator
		"""
		BaselineEvaluator.__init__(self, *args, **kwargs)
		self.filePrefix = ""
		self.system_id_list = []

	def getOptimumParameters(self, f, optPDict, latent_space_size):
		self._setmetricString ()
		
		for lambda_ in  self.lambda_list:
			Logger.logr.info("[CON-S2V-C] Starting running with lambda = %s" %(lambda_))
			os.environ["NBR_TYPE"]=str(0)
			os.environ["FULL_DATA"]=str(0)
			os.environ["LAMBDA"]=str(lambda_)
			jointL = JointLearningSen2VecRunner(self.dbstring)
			jointL.window = optPDict["window"]
			if lambda_== self.lambda_list[0]:
			   	jointL.prepareData(1)
			self.metric[lambda_] = self.evaluate(jointL, self.filePrefix, latent_space_size)
			Logger.logr.info("[CON-S2V-C Baseline] %s for lambda %.2f = %s"\
			 	%(self.metric_str, lambda_, self.metric[lambda_]))

		joint_lambda_opt = max(self.metric, key=self.metric.get) 
		optPDict['con-s2v-c-lambda'] = joint_lambda_opt
		f.write("[CON-S2V-C Baseline] Optimal lambda : %.2f%s" %(joint_lambda_opt, os.linesep))
		f.write("[CON-S2V-C Baseline] %ss: %s%s" %(self.metric_str, self.metric, os.linesep))
		f.flush()
		
		return optPDict

	def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
		os.environ["NBR_TYPE"] = str(0)
		os.environ["FULL_DATA"] = str(0)
		os.environ["LAMBDA"] =  str(optPDict['con-s2v-c-lambda'])
		jointL = JointLearningSen2VecRunner(self.dbstring)	
		jointL.window = optPDict["window"]
		self.system_id_list.append(jointL.system_id)
		self.writeResults(pd, rbase, latent_space_size, jointL, self.filePrefix, f)
