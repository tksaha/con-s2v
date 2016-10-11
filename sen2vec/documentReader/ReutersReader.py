#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
import operator
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from bs4 import BeautifulSoup
from log_manager.log_config import Logger 
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner
from baselineRunner.P2VSENTCExecutableRunner import P2VSENTCExecutableRunner
from baselineRunner.RegularizedSen2VecRunner import RegularizedSen2VecRunner

from evaluation.rankingevaluation.RankingEvaluation import RankingEvaluation 


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
		
		self.postgres_recorder.insertIntoDocTopTable(document_id,\
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
		are assigned for a particular document. We take top-10 
		frequent topics mentioned in "Text Categorization with support 
		vector machines: Learning with many relevant features."
		"""
		interested_topic_list = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade'\
			,'interest', 'ship', 'wheat', 'corn']

		topics = doc.find("topics").findAll('d')
		for topic in topics: 
			topic = topic.text.strip() 
			if topic in interested_topic_list: 
				return topic; 

		return "other"


	def readDocument(self, ld):

		"""
		First, reading and recording the Topics. Second, recording each document at a time	
		Third, for each document, record the lower level information 
		like: paragraph, sentences in table 
		"""

		if ld <= 0: return 0 
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.alterSequences()
		self.readTopic() 
		
		
		for file_ in os.listdir(self.folderPath):
			if file_.endswith(".sgm"):
				file_content = self._getTextFromFile("%s%s%s" %(self.folderPath,"/",file_))
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

					topic = self._getTopic(document_id, doc)
					

					if topic in ['wheat', 'corn', 'other']:
					 	continue
						
					self.postgres_recorder.insertIntoDocTable(document_id, title, \
								doc_content, file_, metadata)
						
					istrain = 'YES' if doc['lewissplit'].lower() == 'train' else 'NO'
					self.__recordDocumentTopic(document_id, doc)
					self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder,topic, istrain)
					
					
		Logger.logr.info("Document reading complete.")
		return 1
	

	def runBaselines(self, pd, rbase, gs):
		"""
		"""
		latent_space_size = 300
	
		# Logger.logr.info("Starting Running Para2vec Baseline")
		# paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
		# paraBaseline.prepareData(pd)
		# paraBaseline.runTheBaseline(rbase,latent_space_size)
		# if gs ==1: self.postgres_recorder.truncateSummaryTable()
		# paraBaseline.generateSummary(gs)
		# paraBaseline.runEvaluationTask()

		# Logger.logr.info("Starting Running Node2vec Baseline")	
		n2vBaseline = Node2VecRunner(self.dbstring)
		# n2vBaseline.prepareData(pd)
		n2vBaseline.runTheBaseline(rbase, latent_space_size)
		# n2vBaseline.generateSummary(gs)
		n2vBaseline.runEvaluationTask()

		# iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
		# iterrunner.prepareData(pd)
		# iterrunner.runTheBaseline(rbase)
		# iterrunner.generateSummary(gs)
		# iterrunner.runEvaluationTask()

# 		# evaluation = RankingEvaluation(['n2v', 'p2v'])
# 		# print (evaluation._getRankingEvaluation())

		# regs2v = RegularizedSen2VecRunner(self.dbstring)
		# regs2v.prepareData(pd)
		# regs2v.runTheBaseline(rbase, latent_space_size)
		# #regs2v.generateSummary(gs)
		# regs2v.runEvaluationTask()
		





