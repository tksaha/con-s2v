#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
import re
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner
from baselineRunner.Node2VecRunner  import Node2VecRunner
from baselineRunner.Paragraph2VecRunner import Paragraph2VecRunner

class IMDBReader(DocumentReader):
	""" 
	IMDB Document Reader. Reads IMDB documents extracted from 
	: 
	"""

	def __init__(self,*args, **kwargs):
		"""
		Initialization assumes that IMDB_PATH environment is set. 
		To set in linux or mac: export IMDB_PATH=/some_directory_containing_IMDB_data
		"""
		DocumentReader.__init__(self, *args, **kwargs)
		self.dbstring = os.environ["IMDB_DBSTRING"]
		self.postgres_recorder = PostgresDataRecorder(self.dbstring)
		self.folderPath = os.environ['IMDB_PATH']
	
	def readTopic(self):
		"""
		"""
		rootDir = "%s/train" %self.folderPath
		return self._getTopics(rootDir)
	
	def readDocument(self, ld): 
		"""
		"""
		if ld <= 0: return 0 	
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.alterSequences()
		topic_names = self.readTopic()
		
		
		document_id = 0
		for first_level_folder in next(os.walk(self.folderPath))[1]:
			if not(DocumentReader._folderISHidden(self, first_level_folder)):
				for topic in topic_names:					
					if first_level_folder == 'test' and topic == 'unsup':
						continue
					for file_ in os.listdir("%s%s%s%s%s" %(self.folderPath, "/", \
											first_level_folder, "/", topic)):
						doc_content = self._getTextFromFile("%s%s%s%s%s%s%s" \
							%(self.folderPath, "/", first_level_folder, "/", topic, "/", file_))
						
						document_id += 1
						title, metadata, istrain = None, None, None					
						try:
							trainortest = first_level_folder
							metadata = "SPLIT:%s"%trainortest
							istrain = 'YES' if trainortest.lower() == 'train' else 'NO'			
						except:
							Logger.logr.info("NO MetaData or Train Test Tag")
						self.postgres_recorder.insertIntoDocTable(document_id, title, \
									doc_content, file_, metadata) 
						category = topic.split('.')[0]
						self.postgres_recorder.insertIntoDocTopTable(document_id, \
									[topic], [category]) 		
						self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder, topic, istrain)
					
					
		Logger.logr.info("Document reading complete.")
		return 1
	
	
	def runBaselines(self):
		"""
		"""
		latent_space_size = 300
		Logger.logr.info("Starting Running Para2vec (Doc) Baseline")
		# paraBaseline = Paragraph2VecSentenceRunner(self.dbstring)
		# paraBaseline.prepareData()
		# paraBaseline.runTheBaseline(latent_space_size)

		# Logger.logr.info("Starting Running Node2vec Baseline")
		# n2vBaseline = Node2VecRunner(self.dbstring)
		# n2vBaseline.prepareData()

		# paraBaseline.runEvaluationTask()
		# paraBaseline.runClassificationTask()
		
#		n2vBaseline.runTheBaseline(latent_space_size)

#		Logger.logr.info("Starting Running Iterative Update Method")
#		iterUdateBaseline = IterativeUpdateRetrofitRunner(self.dbstring)
#		iterUdateBaseline.prepareData()
#		iterUdateBaseline.runTheBaseline()
		
		docBaseLine = Paragraph2VecRunner(self.dbstring)
		#docBaseLine.prepareData()
		#docBaseLine.runTheBaseline(latent_space_size)
		docBaseLine.runEvaluationTask()
		docBaseLine.runClassificationTask()

	
