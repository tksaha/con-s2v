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

from evaluation.rankingevaluation.RankingEvaluation import RankingEvaluation 

# There are some summaries [ex:fbis4-45908, FT932-15960] for which the 
# original document is not present
class DUCReader(DocumentReader):
	""" 
	DUC Document Reader

	"""

	def __init__(self,*args, **kwargs):
		"""
		It reads he environment variable and initializes the 
		base class. 
		"""
		DocumentReader.__init__(self, *args, **kwargs)
		self.dbstring = os.environ["DUC_DBSTRING"]
		self.postgres_recorder = PostgresDataRecorder(self.dbstring)
		self.folderPath = os.environ['DUC_PATH']
		self.processed_filenames = []
		self.processed_summaries = []

	def readTopic(self):
		"""
		Recording DUC years as topics.
		"""
		topic_names = ['2001', '2002', '2003', '2004', '2005', '2006', '2007']
		categories = topic_names
		self.postgres_recorder.insertIntoTopTable(topic_names, categories)
		Logger.logr.info("Topic reading complete.")
	
	def recordFirstSentenceBaselineSummary(self, document_id):
		"""
		Recording First Sentence as a Baseline Summary
		"""
		sentence_id = self.postgres_recorder.selectFirstSentBaselineId(document_id)
		method_id = 21 #First sentence as a baseline summary= 21
		position = 1
		self.postgres_recorder.insertIntoSumTable(document_id, method_id, sentence_id, position)
	
	def recordDocuments(self, documents, document_id, topic):
		for document in documents:
			filename = document.split(os.path.sep)[-1] #ft923-5089
			if filename in self.processed_filenames: #don't store duplicate files
				continue
			self.processed_filenames += [filename]
			document_id += 1
			doc_content = self._getTextFromFile("%s" %(document))
			title, metadata, istrain = None, None, 'YES'
			soup = BeautifulSoup(doc_content, "html.parser")
			try:
				doc_content = soup.find('text').text
			except:
#				Logger.logr.info("%s %s" %(document, "Skipping. Cause, TEXT tag not found"))
				continue
			self.postgres_recorder.insertIntoDocTable(document_id, title, \
						doc_content, filename, metadata) 
			category = topic.split('.')[0]
			self.postgres_recorder.insertIntoDocTopTable(document_id, \
						[topic], [category])
			self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder, topic, istrain)
			
			self.recordFirstSentenceBaselineSummary(document_id) #Recording First Sentence as a Baseline Summary
		return document_id
		
	def recordSummaries(self, summaries):
		for summary in summaries:
			doc_content = self._getTextFromFile("%s" %(summary))
			soup = BeautifulSoup(doc_content, "html.parser")
			sums = soup.findAll('sum')
			for sum_ in sums:
				filename = sum_.get('docref')
				doc_content = sum_.text
				method_id = 20 #DUC = 20
				summarizer = sum_.get('summarizer')
				metadata = "SUMMARIZER:%s" %(summarizer)
				if "%s%s" %(filename, summarizer) in self.processed_summaries:
					continue
				self.processed_summaries += ["%s%s" %(filename, summarizer)]
				self.postgres_recorder.insertIntoGoldSumTable(filename, doc_content, \
							method_id, metadata)

	def _readDUC2001(self, document_id):
		"""
		It loads the DUC 2001 documents into
		the database
		"""
		topic = "2001"
		cur_path = "%s/%s" %(self.folderPath, "DUC2001")
		summaries = []
		documents = []
		for root, directories, files in os.walk(cur_path):
			for file_ in files:
				if file_ in ['50', '100', '200', '400']:
					pass
				elif file_ in ['perdocs']:
					summaries += [os.path.join(root, file_)]
				else:
					documents += [os.path.join(root, file_)]
		
		Logger.logr.info("Recording DUC 2001 Documents.")
		document_id = self.recordDocuments(documents, document_id, topic)
		
		Logger.logr.info("Recording DUC 2001 Summaries.")
		self.recordSummaries(summaries)
		
		return document_id
	
	def _readDUC2002(self, document_id):
		"""
		It loads the DUC 2002 documents into
		the database
		"""
		topic = "2002"
		cur_path = "%s/%s/%s" %(self.folderPath, "DUC2002", "docs")
		documents = []

		for root, directories, files in os.walk(cur_path):
			documents += [os.path.join(root, file_) for file_ in files]
		
		Logger.logr.info("Recording DUC 2002 Documents.")
		document_id = self.recordDocuments(documents, document_id, topic)
				
		cur_path = "%s/%s/%s" %(self.folderPath, "DUC2002", "summaries")
		summaries = []
		for root, directories, files in os.walk(cur_path):
			summaries += [os.path.join(root, file_) for file_ in files if file_.endswith('perdocs')]
		
		Logger.logr.info("Recording DUC 2002 Summaries.")
		self.recordSummaries(summaries)
		
		return document_id
		
	def _readDUC2003(self, document_id):
		"""
		It loads the DUC 2003 documents into
		the database
		"""
		topic = "2003"
		cur_path = "%s/%s/%s" %(self.folderPath, "DUC2003", "duc2003_testdata")
		documents = []

		for root, directories, files in os.walk(cur_path):
			documents += [os.path.join(root, file_) for file_ in files]
		
		Logger.logr.info("Recording DUC 2003 Documents.")
		document_id = self.recordDocuments(documents, document_id,topic)
				
		cur_path = "%s/%s/%s/%s" %(self.folderPath, "DUC2003", "detagged.duc2003.abstracts", "models")
		summaries = []
		for root, directories, files in os.walk(cur_path):
			summaries += [os.path.join(root, file_) for file_ in files if file_.split('.')[1] == 'P']
		
		Logger.logr.info("Recording DUC 2003 Summaries.")
		for summary in summaries:
			doc_content = self._getTextFromFile("%s" %(summary))
			filename = '.'.join(summary.split(os.path.sep)[-1].split('.')[5:7])
			method_id = 20 #DUC = 20
			summarizer = summary.split(os.path.sep)[-1].split('.')[4]
			metadata = "SUMMARIZER:%s" %(summarizer)
			if "%s%s" %(filename, summarizer) in self.processed_summaries:
				continue
			self.processed_summaries += ["%s%s" %(filename, summarizer)]
			self.postgres_recorder.insertIntoGoldSumTable(filename, doc_content, \
							method_id, metadata)
		return document_id
		
	def readDocument(self, ld):	
		if ld <= 0: return 0 
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.truncateSummaryTable()
		self.postgres_recorder.alterSequences()
		self.readTopic()
		
		document_id = 0
		document_id = self._readDUC2001(document_id)
		document_id = self._readDUC2002(document_id)
#		document_id = self._readDUC2003(document_id)
		
	def runBaselines(self, pd, rbase, gs):
		"""
		"""
#		latent_space_size = 300
#	
#		Logger.logr.info("Starting Running Para2vec Baseline")
#		paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
#		paraBaseline.prepareData(pd)
#		paraBaseline.runTheBaseline(rbase,latent_space_size)
##		if gs ==1: self.postgres_recorder.truncateSummaryTable()
#		paraBaseline.generateSummary(gs)
		#paraBaseline.runEvaluationTask()

#		Logger.logr.info("Starting Running Node2vec Baseline")	
#		n2vBaseline = Node2VecRunner(self.dbstring)
#		n2vBaseline.prepareData(pd)
#		n2vBaseline.runTheBaseline(rbase, latent_space_size)
#		n2vBaseline.generateSummary(gs)
#		n2vBaseline.runEvaluationTask()

#		iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
#		iterrunner.prepareData(pd)
#		iterrunner.runTheBaseline(rbase)
#		iterrunner.generateSummary(gs)
#		iterrunner.runEvaluationTask()

		evaluation = RankingEvaluation(models = [20], systems = [1, 2, 21])
		evaluation._getRankingEvaluation()
