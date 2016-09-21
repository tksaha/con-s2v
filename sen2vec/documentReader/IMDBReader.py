#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
import re


class IMDBReader(DocumentReader):
	""" 
	IMDB Document Reader.
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
	
	
	def __get_text_from_file(self, file):
		"""

		"""
		text = ""
		with open(file, encoding='utf-8', errors='ignore') as f:
			for line in f:
				text = "%s%s" %(text, line)
		return text
	
	
	def readTopic(self):
		"""
		"""
		first_level_folder =  "train"
		folder_names = next(os.walk("%s%s%s" %(self.folderPath, "/", first_level_folder)))[1]
		topic_names = [name for name in folder_names if not (DocumentReader._folder_is_hidden(self, name))]
		categories = [name.split('.')[0] for name in topic_names]

		self.postgres_recorder.insertIntoTopTable(topic_names, categories)						
		Logger.logr.info("Topic reading complete.")
		return topic_names
	
	def readDocument(self, ld): 
		"""
		"""
		if ld <= 0:
			return 0 
			
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.altersequences()

		topic_names = self.readTopic()
		
		document_id = 0
		for first_level_folder in next(os.walk(self.folderPath))[1]:
			if not(DocumentReader._folder_is_hidden(self, first_level_folder)):
				for topic in topic_names:					
					if first_level_folder == 'test' and topic == 'unsup':
						continue
					for file_ in os.listdir("%s%s%s%s%s" %(self.folderPath, "/", \
											first_level_folder, "/", topic)):
						doc_content = self.__get_text_from_file("%s%s%s%s%s%s%s" \
							%(self.folderPath, "/", first_level_folder, "/", topic, "/", file_))
						
						document_id += 1
						title = None						

						metadata = None
						istrain = None 
						try:
							trainortest = first_level_folder
							metadata = "SPLIT:%s"%trainortest
							if trainortest.lower() == 'train':
								istrain = 'YES'
							else:
								istrain = 'NO'
						except:
							pass 

						self.postgres_recorder.insertIntoDocTable(document_id, title, \
									doc_content, file_, metadata) 

						category = topic.split('.')[0]
						self.postgres_recorder.insertIntoDoc_TopTable(document_id, \
									[topic], [category]) 		
						self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder, topic, istrain)
					
					
		Logger.logr.info("Document reading complete.")
		return 1
	
	
	def runBaselines(self):
		"""
		"""
		pass
