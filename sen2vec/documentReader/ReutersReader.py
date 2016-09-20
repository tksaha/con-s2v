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
		It reads he environment variable and initializes the 
		base class. 
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
				"exchanges", "companies"] 

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



	def _getTopic(self, document_id, doc):
		"""
		Interested topic: acq, money-fx, crude, trade, interest. 
		A topic can be one of the interested topic. A topic 
		is assigned based on the order if multiple interested topics 
		are assigned for a particular document. 
		"""
		interested_topic_list = ['acq', 'money-fx', 'crude', 'trade', 'interest']

		topics = doc.find("topics").findAll('d')
		for topic in topics: 
			if topic in interested_topic_list: 
				return topic; 

		return "other"


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

						if doc['lewissplit'] == "NOT-USED" or doc['topics'] == "NO"\
						or doc['topics'] == "BYPASS" :
							Logger.logr.info("SKipping because of ModApte Split")
							continue
					except:
						metadata = None
						continue 

					self.postgres_recorder.insertIntoDocTable(document_id, title, \
								doc_content, file_, metadata) 

					topic = self._getTopic(document_id, doc)
					if doc['lewissplit'].lower() == 'train':
					   istrain = 'Yes'
					else:
					   istrain = 'NO'
					   
					self.__recordDocumentTopic(document_id, doc)			
					self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder,topic, istrain)
					
					
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
		# n2vBaseline.runTheBaseline(latent_space_size)

		# Logger.logr.info("Starting Running Iterative Update Method")
		# iterUdateBaseline = IterativeUpdateRetrofitRunner(self.dbstring)
		# iterUdateBaseline.prepareData()
		# iterUdateBaseline.runTheBaseline()





