#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
import re
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner
from baselineRunner.Node2VecRunner  import Node2VecRunner
from baselineRunner.Paragraph2VecRunner import Paragraph2VecRunner
from baselineRunner.Paragraph2VecCEXERunner import Paragraph2VecCEXERunner


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


	def __readAPass(self, load=0):
		train_doc_ids = []
		
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
							if trainortest.lower() == 'train' and topic !='unsup':
							   train_doc_ids.append(document_id)
						except:
							Logger.logr.info("NO MetaData or Train Test Tag")
	
	def readDocument(self, ld): 
		"""
		"""
		if ld <= 0: return 0            
        self.postgres_recorder.trucateTables()
        self.postgres_recorder.alterSequences()

        train_doc_ids = self.__readAPass(0)
        print (len(train_doc_ids))
        #self.__createValidationSet(train_doc_ids)
        #self.__readAPass(1)
        			
		Logger.logr.info("Document reading complete.")
		return 1
	
	
	def runBaselines(self):
		"""
		"""
		latent_space_size = 300
	
