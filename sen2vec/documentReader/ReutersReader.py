#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import operator
import numpy as np 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from bs4 import BeautifulSoup
from log_manager.log_config import Logger 
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner
from baselineRunner.P2VSENTCExecutableRunner import P2VSENTCExecutableRunner
from baselineRunner.RegularizedSen2VecRunner import RegularizedSen2VecRunner
from baselineRunner.DictRegularizedSen2VecRunner import DictRegularizedSen2VecRunner
from baselineRunner.JointLearningSen2VecRunner import JointLearningSen2VecRunner

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
		self.validationDict = {}


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

	def __createValidationSet(self, document_ids):

		total_doc = len(document_ids)
		nvalid_doc = float(total_doc * 0.20)

		np.random.seed(2000)
		valid_list = np.random.choice(document_ids, nvalid_doc, replace=False).tolist()

		for id_ in valid_list:
			self.validationDict[id_] = 1

	def __readAPass(self,load):
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.alterSequences()
		self.readTopic() 
		
		train_doc_ids = []
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
					#if topic not in ['ship','interest']:
					#	continue				
						
					istrain = 'YES' if doc['lewissplit'].lower() == 'train' else 'NO'
					if document_id in self.validationDict:
						istrain ='VALID'

					if istrain == 'YES':
						train_doc_ids.append(document_id)

					if  load==0:
						continue 
					self.postgres_recorder.insertIntoDocTable(document_id, title, \
								doc_content, file_, metadata)
					self.__recordDocumentTopic(document_id, doc)
					self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder,topic, istrain)
					
		return train_doc_ids			
		Logger.logr.info("[Pass 1] Document reading complete.")

	def readDocument(self, ld):

		"""
		First, reading and recording the Topics. Second, recording each document at a time	
		Third, for each document, record the lower level information 
		like: paragraph, sentences in table 
		"""
		if ld <= 0: return 0 
		train_doc_ids = self.__readAPass(0)
		self.__createValidationSet(train_doc_ids)
		self.__readAPass(1)
		return 1
		

	def runBaselines(self, pd, rbase, gs):
		"""
		"""
		# optDict = self._runClassificationOnValidation(pd, rbase, gs,"reuter")
		# self.doTesting(optDict, "reuter", rbase, pd, gs, True)
		# optDict = self._runClusteringOnValidation(pd, rbase, gs, "reuter")
		# self.doTesting(optDict, "reuter", rbase, pd, gs, False)

		os.environ['EVAL'] = 'TEST'
		os.environ['TEST_FOR'] = 'CLASS'
		jointL = JointLearningSen2VecRunner(self.dbstring)
		jointL.prepareData(pd)
		jointL.runTheBaseline(rbase, 300)
		jointL.runEvaluationTask()