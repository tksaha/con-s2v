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
	
	def recordFirstSentenceBaselineSummary(self, filename, doc_content):
		"""
		Recording First Sentence as a Baseline Summary
		"""
		if len(doc_content) > 0:
			baseline_content = self._splitIntoSentences(doc_content)[0]
		else:
			baseline_content = ""
		method_id = 21 #First sentence as a baseline summary= 21
		metadata = "SUMMARIZER:%s" %("FirstLine")
		self.postgres_recorder.insertIntoGoldSumTable(filename, baseline_content, \
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
			summaries += [os.path.join(root, file_) for file_ in files if file_.endswith('.abs')]
			documents += [os.path.join(root, file_) for file_ in files if file_.endswith('.body')]
		
		Logger.logr.info("Recording DUC 2001 Documents.")
		for document in documents:
			filename = document.split(os.path.sep)[-1].split('.')[0] #ft923-5089
			if filename in self.processed_filenames: #don't store duplicate files
				continue
			self.processed_filenames += [filename]
			document_id += 1
			doc_content = self._getTextFromFile("%s" %(document))
			title, metadata, istrain = None, None, 'YES' 
			self.postgres_recorder.insertIntoDocTable(document_id, title, \
						doc_content, filename, metadata) 
			category = topic.split('.')[0]
			self.postgres_recorder.insertIntoDocTopTable(document_id, \
						[topic], [category])
			self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder, topic, istrain)
			
			self.recordFirstSentenceBaselineSummary(filename, doc_content) #Recording First Sentence as a Baseline Summary
		
		Logger.logr.info("Recording DUC 2001 Summaries.")
		for summary in summaries:
			filename = summary.split(os.path.sep)[-1].split('.')[0] #ft923-5089
			doc_content = self._getTextFromFile("%s" %(summary))
			method_id = 20 #DUC = 20
			summarizer = summary.split(os.path.sep)[-3][-1]
			metadata = "SUMMARIZER:%s" %(summarizer) #for example: f of d32f
			if "%s%s" %(filename, summarizer) in self.processed_summaries:
				continue
			self.processed_summaries += ["%s%s" %(filename, summarizer)]
			
			self.postgres_recorder.insertIntoGoldSumTable(filename, doc_content, \
						method_id, metadata)
		
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
		for document in documents:
			filename = document.split(os.path.sep)[-1] #LA012190-0194
			if filename in self.processed_filenames: #don't store duplicate files
				continue
			self.processed_filenames += [filename]
			document_id += 1
			doc_content = self._getTextFromFile("%s" %(document))
			title, metadata, istrain = None, None, 'YES' 
			soup = BeautifulSoup(doc_content, "html.parser")
			doc_content = soup.find('text').text
			self.postgres_recorder.insertIntoDocTable(document_id, title, \
						doc_content, filename, metadata) 
			category = topic.split('.')[0]
			self.postgres_recorder.insertIntoDocTopTable(document_id, \
						[topic], [category])

			self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder, topic, istrain)
			
			self.recordFirstSentenceBaselineSummary(filename, doc_content) #Recording First Sentence as a Baseline Summary
				
		cur_path = "%s/%s/%s" %(self.folderPath, "DUC2002", "summaries")
		summaries = []
		for root, directories, files in os.walk(cur_path):
			summaries += [os.path.join(root, file_) for file_ in files if file_.endswith('perdocs')]
		
		Logger.logr.info("Recording DUC 2002 Summaries.")
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
		for document in documents:
			filename = document.split(os.path.sep)[-1] #LA012190-0194
			if filename in self.processed_filenames: #don't store duplicate files
				continue
			
			doc_content = self._getTextFromFile("%s" %(document))
			title, metadata, istrain = None, None, 'YES' 
			soup = BeautifulSoup(doc_content, "html.parser")
			try:
				doc_content = soup.find('text').text
			except:
#				Logger.logr.info("%s %s" %(document, "Skipping. Cause, TEXT tag not found"))
				continue
			
			self.processed_filenames += [filename]
			document_id += 1
			self.postgres_recorder.insertIntoDocTable(document_id, title, \
						doc_content, filename, metadata) 
			category = topic.split('.')[0]
			self.postgres_recorder.insertIntoDocTopTable(document_id, \
						[topic], [category])

			self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder, topic, istrain)
			
			self.recordFirstSentenceBaselineSummary(filename, doc_content) #Recording First Sentence as a Baseline Summary
				
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
		self.postgres_recorder.alterSequences()
		self.readTopic()
		
		document_id = 0
		document_id = self._readDUC2001(document_id)
		document_id = self._readDUC2002(document_id)
		document_id = self._readDUC2003(document_id)
		
	def runBaselines(self, pd, rbase, gs):
		"""
		"""
		pass
