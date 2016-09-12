#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from bs4 import BeautifulSoup
from log_manager.log_config import Logger 
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner


class ReutersReader(DocumentReader):
	""" 
	Reuters Document Reader

	"""

	def __init__(self,*args, **kwargs):
		"""
		"""
		DocumentReader.__init__(self, *args, **kwargs)
		self.dbstring = os.environ["REUTERS_DBSTRING"]
		self.postgres_recorder = PostgresDataRecorder(self.dbstring)
		self.folderPath = os.environ['REUTERS_PATH']


	def __recordDocumentTopic (self, document_id, doc):
		"""

		"""
		topic_names = []
		categories = []
							
		possible_categories = ["topics", "places", "people", "orgs", 
				"exchanges", "companies"] # List of possible topics

		for category in possible_categories:
			try:
				topics = doc.find(category).findAll('d')
				for topic in topics:
					topic = topic.text.strip()
					topic_names += [topic]
					categories += [category]					
			except:
				pass
		
		self.postgres_recorder.insertIntoDoc_TopTable(document_id,\
					topic_names, categories) 


	def __recordParagraphAndSentence(self, document_id, doc_content):
		"""
		"""
		paragraphs = self._splitIntoParagraphs(doc_content)

		for position, paragraph in enumerate(paragraphs):
			paragraph_id = self.postgres_recorder.insertIntoParTable(paragraph)
			self.postgres_recorder.insertIntoDoc_ParTable(document_id, paragraph_id, position)
			
			sentences = self._splitIntoSentences(paragraph)
			for sentence_position, sentence in enumerate(sentences):
				sentence_id = self.postgres_recorder.insertIntoSenTable(sentence)
				self.postgres_recorder.insertIntoPar_SenTable(paragraph_id, sentence_id,\
					sentence_position)
		

	def readTopic(self):
		"""
		"""
		topic_names = []
		categories = []
		for file_ in os.listdir(self.folderPath):
			if file_.endswith(".lc.txt"):
				category = file_.split('-')[1]
				content = open("%s%s%s" %(self.folderPath,"/",file_), 'r', 
					encoding='utf-8', errors='ignore').read()
				for topic in content.split(os.linesep):
					topic = topic.strip()
					if len(topic) != 0:
						topic_names += [topic]
						categories += [category]

		self.postgres_recorder.insertIntoTopTable(topic_names, categories)						
		Logger.logr.info("Topic reading complete.")


	def readDocument(self, ld):

		"""
		First, reading and recording the Topics
		Second, recording each document at a time	
		Third, for each document, record the lower level information 
		like: paragraph, sentences in table 
		"""

		if ld <= 0:
			return 0 
			
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.altersequences()

		self.readTopic() 
		
		
		for file_ in os.listdir(self.folderPath):
			if file_.endswith(".sgm"):
				file_content = open("%s%s%s" %(self.folderPath,"/",file_), 'r', 
					encoding='utf-8', errors='ignore').read()
				soup = BeautifulSoup(file_content, "html.parser")

				for doc in soup.findAll('reuters'):
					document_id = doc['newid']
					
					title = doc.find('title').text if doc.find('title') \
								is not None else None 
					doc_content = doc.find('text').text if doc.find('text')\
							 is not None else None 

					try:
						metadata = "OLDID:"+doc['oldid']+"^"+"TOPICS:"+doc['topics']+\
						"^"+"CGISPLIT:"+doc['cgisplit']+"^"+"LEWISSPLIT:"+doc['lewissplit']
					except:
						metadata = None
					
					self.postgres_recorder.insertIntoDocTable(document_id, title, \
								doc_content, file_, metadata) 


					self.__recordDocumentTopic(document_id, doc)			
					self.__recordParagraphAndSentence(document_id, doc_content)
					
					
		Logger.logr.info("Document reading complete.")
		return 1

	def runBaselines(self):
		"""
		"""
		latent_space_size = 128
		Logger.logr.info("Starting Running Para2vec Baseline")
		paraBaseline = Paragraph2VecSentenceRunner(self.dbstring)
		paraBaseline.prepareData()
		paraBaseline.runTheBaseline(latent_space_size)

		Logger.logr.info("Starting Running Node2vec Baseline")
		n2vBaseline = Node2VecRunner(self.dbstring)
		n2vBaseline.prepareData()
		n2vBaseline.runTheBaseline(latent_space_size)

		Logger.logr.info("Starting Running Iterative Update Method")
		iterUdateBaseline = IterativeUpdateRetrofitRunner(self.dbstring)
		iterUdateBaseline.prepareData()
		iterUdateBaseline.runTheBaseline()





