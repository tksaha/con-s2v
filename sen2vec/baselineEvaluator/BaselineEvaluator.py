#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
from rouge.Rouge import Rouge 
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger

class BaselineEvaluator:
	"""
	"""
	def __init__(self, dbstring, **kwargs):
		self.dbstring = dbstring
		self.window_size_list = ["8", "10", "12"]
		self.beta_list = [0.3, 0.5, 0.8, 1.0]
		self.metric = {}
		self.metric_str = ""

	def _setmetricString (self):
		self.metric = {}
		self.metric_str = "F1"
		if os.environ['VALID_FOR'] == 'CLUST':
		   self.metric_str = "AdjMIScore"
		elif os.environ['VALID_FOR'] == 'RANK':
		   self.metric_str = "Recall"

	def _getAdjustedMutulScore(self, latreprName):
		file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_rawclustereval_2.txt"%latreprName)
		for line in open(file_):
			if "Adjusted Mutual Info Score:" in line:
				adj_score = line.strip()[line.strip().rfind(":")+1:]
				adj_score = float(adj_score)
				return adj_score

	def _getF1(self, latreprName):
		"""
		"""
		file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_raweval_2.txt"%latreprName)
		for line in open(file_):
			if "avg" in line:
				line_elems = line.strip().split()
				f1 = float(line_elems[5])
				return f1 

	 def __runSpecificEvaluation(self, models = [20], systems = []):
        rougeInstance = Rouge()
        rPDict = rougeInstance.buildRougeParamDict()
        rPDict['-l'] = str(100)
        rPDict['-c'] = str(0.99)

        evaluation = RankingEvaluation(topics = [self.duc_topic], models = models, systems = systems)
        evaluation._prepareFiles()
        evaluation._getRankingEvaluation(rPDict, rougeInstance)

	def _writeResult(self, latreprName, f):
		if os.environ['TEST_FOR'] == 'CLASS':
			file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_raweval_2.txt"%latreprName)
			for line in open(file_):
				f.write(line)
		else:
			file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_rawclustereval_2.txt"%latreprName)
			for line in open(file_):
				f.write(line)

	def __getRecall(self, method_id, models, systems):
        output_file_name = ""
        for model in models:
            output_file_name += str(model)+"_"
        for system in systems:
            output_file_name += "_"+str(system)
        output_file_name += "_output"
        output_file_name += "_%s.txt" %(str(100))
        
        with open('%s%s%s' %(os.environ["SUMMARYFOLDER"],"/",output_file_name), 'r') as f:
            content = f.read()
            recall = float(content.split("%s ROUGE-1 Average_R: " %method_id)[1].split(' ')[0])
        return recall

	def evaluate(self, baseline, prefix, latent_space_size):
		if os.environ['VALID_FOR'] == 'CLASS':
			baseline.runTheBaseline(1, latent_space_size)
			baseline.runEvaluationTask()
			f1 = self._getF1("%s%s"%(baseline.latReprName, prefix))
			return f1
		elif os.environ['VALID_FOR'] == 'CLUST':
			baseline.runTheBaseline(1, latent_space_size)
			baseline.runEvaluationTask()
			adj = self._getAdjustedMutulScore("%s%s"%(baseline.latReprName, prefix))
			return adj
		else:
			baseline.runTheBaseline(1,latent_space_size, window)
            baseline.generateSummary(1,method_id,"",\
                         lambda_val=1.0, diversity=False)
            baseline.doHouseKeeping()           
            self.__runSpecificEvaluation(models = [20], systems = [baseline.system_id]) 
            return self.__getRecall(method_id=baseline.system_id,\
            	 models = [20], systems = [baseline.system_id])

	def writeResults(self, pd, rbase, latent_space_size, baseline, filePrefix, f):

		if os.environ['TEST_FOR'] == 'RANK':
			baseline.prepareData(pd)      
            baseline.runTheBaseline(rbase,latent_space_size)
            baseline.generateSummary(gs,method_id,"",\
                         lambda_val=1.0, diversity=False)
            baseline.doHouseKeeping()  
		else:
			baseline.prepareData(pd)		
			baseline.runTheBaseline(rbase,latent_space_size)
			baseline.runEvaluationTask()
			self._writeResult("%s_%s"%(baseline.latReprName, filePrefix), f)
			baseline.doHouseKeeping()	
			f.flush()