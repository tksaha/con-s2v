#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
import re
import numpy as np 
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner
from baselineRunner.P2VSENTCExecutableRunner import P2VSENTCExecutableRunner
from baselineRunner.JointSupervisedRunner import JointSupervisedRunner
from baselineRunner.JointLearningSen2VecRunner import JointLearningSen2VecRunner
from baselineRunner.FastSentVariantRunner import FastSentVariantRunner
from evaluation.rankingevaluation.RankingEvaluation import RankingEvaluation

class NewsGroupReader(DocumentReader):
	""" 
	News Group Document Reader.
	"""

	def __init__(self,*args, **kwargs):
		"""
		Initialization assumes that NEWSGROUP_PATH environment is set. 
		To set in linux or mac: export NEWSGROUP_PATH=/some_directory_containing_newsgroup_data
		"""
		DocumentReader.__init__(self, *args, **kwargs)
		self.dbstring = os.environ["SICK_DBSTRING"]
		self.postgres_recorder = PostgresDataRecorder(self.dbstring)
		self.folderPath = os.environ['SICK_PATH']
		self.validation_pairs = {}
		self.test_pairs = {}


	
	def readDocument(self, ld): 
		"""
		Stripping is by default inactive. For future reference it has been 
		imported from scikit-learn newsgroup reader package. 

		
		"""
		if ld <= 0: return 0 			
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.alterSequences()

		# Read SICK Data
		sick_file = open(os.path.join(self.folderPath, ))
		
		return 1
	
	
	def runBaselines(self, pd, rbase, gs):
		"""
		"""
		