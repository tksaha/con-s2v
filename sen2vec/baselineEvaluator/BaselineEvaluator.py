#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
from abc import ABCMeta, abstractmethod


class BaselineEvaluator:
	"""
	"""
	def __init__(self, dbstring, **kwargs):
		self.dbstring = dbstring
		self.window_size_list = ["8", "10", "12"]


	def _getAdjustedMutulScore(self, latreprName):
		file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_rawclustereval_2.txt"%latreprName)
		for line in open(file_):
			if "Adjusted Mutual Info Score:" in line:
				adj_score = line.strip()[line.strip().rfind(":")+1:]
				adj_score = float(adj_score)
				Logger.logr.info("Returning value %.2f"%adj_score)
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

	def _writeResult(self, latreprName, f):
		if os.environ['TEST_FOR'] == 'CLASS':
			file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_raweval_2.txt"%latreprName)
			for line in open(file_):
				f.write(line)
		else:
			file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_rawclustereval_2.txt"%latreprName)
			for line in open(file_):
				f.write(line)

	def evaluate(self, fhBaseline, latent_space_size):
		if os.environ['VALID_FOR'] == 'CLASS':
			fhBaseline.runTheBaseline(1, latent_space_size)
			fhBaseline.runEvaluationTask()
			fhBaseline.doHouseKeeping()	
			f1 = self._getF1(fhBaseline.latReprName)
			return f1
		else:
			fhBaseline.runTheBaseline(1, latent_space_size)
			fhBaseline.runEvaluationTask()
			fhBaseline.doHouseKeeping()	
			adj = self._getAdjustedMutulScore(fhBaseline.latReprName)
			return adj

	def writeResults(self, pd, rbase, latent_space_size, fhBaseline, f):
		fhBaseline.prepareData(pd)		
		fhBaseline.runTheBaseline(rbase,latent_space_size)
		fhBaseline.runEvaluationTask()
		self._writeResult("%s"%fhBaseline.latReprName, f)
		fhBaseline.doHouseKeeping()	
		f.flush()
