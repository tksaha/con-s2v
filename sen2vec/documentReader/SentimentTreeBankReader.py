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
from baselineRunner.Paragraph2VecCEXERunner import Paragraph2VecCEXERunner


class SentimentTreeBank2WayReader(DocumentReader):
	def __init__(self, *args, **kwargs):
		"""
		Initialization assumes that SENTTREE_PATH environment is set. 
		"""
		DocumentReader.__init__(self, *args, **kwargs)
		self.dbstring = os.environ["SENTTREE_DBSTRING"]
		self.postgres_recorder = PostgresDataRecorder(self.dbstring)
		self.folderPath = os.environ['SENTTREE_PATH']

	def readTopic(self):
		topic_names = ['pos', 'neg','unsup']
		categories = ['pos', 'neg', 'unsup']

		self.postgres_recorder.insertIntoTopTable(topic_names, categories)				
		Logger.logr.info("[%i] Topic reading complete." %(len(topic_names)))
		return topic_names

	def readDSplit(self,fileName):
		"""
		1 Train, 2 Test, 3 dev
		"""
		line_count = 0 
		dSPlitDict = {}
		for line in open(fileName, encoding='utf-8', errors='ignore'):
			if line_count == 0: 
				pass
			else:	
				doc_id,_, splitid = line.strip().partition(",")
				dSPlitDict[int(doc_id)] = int(splitid)
			line_count = line_count + 1

		Logger.logr.info("Finished reading %i sentences and their splits"%line_count)

		return dSPlitDict;

	def readSentences(self,fileName):
		line_count = 0
		sentenceDict = {}
		for line in open(fileName, encoding='utf-8', errors='ignore'):
			if line_count == 0:
				pass
			else:		
				doc_id,_,sentence = line.strip().partition("\t")
				sentenceDict[int(doc_id)] = sentence.strip()
			line_count = line_count + 1
		return sentenceDict
		Logger.logr.info("Finished reading %i sentence"%line_count)

	def phraseToSentiment(self, fileName):
		line_count = 0 
		phraseToSentimentDict = {}

		for line in open(fileName, encoding='utf-8', errors='ignore'):
			if line_count == 0:
				pass
			else:
				phrase_id,_, sentiment = line.strip().partition("|")
				phraseToSentimentDict[int(phrase_id)] = float(sentiment)
			line_count = line_count + 1
		return phraseToSentimentDict
		Logger.logr.info("Finished reading %i phrases"%line_count)

	def getTopicCategory(self, sentiment_val):
		"""
		[0, 0.2] very negative 
		(0.2, 0.4] negative 
		(0.4, 0.6] neutral 
		(0.6, 0.8] positive 
		(0.8, 1.0] very positive
		"""
		if sentiment_val <=0.4: 
			return ('neg', 'neg')
		elif sentiment_val >0.6:
			return ('pos', 'pos')
		else:
			return ('unsup', 'unsup')

	def readDocument(self, ld): 
		"""
		SKip neutral phrases 
		"""

		if ld <= 0: return 0 			
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.alterSequences()
		topic_names = self.readTopic()

		allPhrasesFile = "%s/dictionary.txt"%(self.folderPath)
		dSPlitDict = self.readDSplit("%s/datasetSplit.txt"%self.folderPath)
		sentenceDict = self.readSentences("%s/datasetSentences.txt"%self.folderPath)
		phraseToSentimentDict = self.phraseToSentiment("%s/sentiment_labels.txt"%self.folderPath)

		for line in open(allPhrasesFile, encoding='utf-8', errors='ignore'):
				phrase, _ , phrase_id = line.strip().partition("|")
				contains_in_train, contains_in_test, contains_in_dev, is_a_sentence = False, False, False, False
				sentiment_val = phraseToSentimentDict[int(phrase_id)]			
				topic, category = self.getTopicCategory(sentiment_val)
				for sent_id, sentence in sentenceDict.items():
					if phrase in sentence: 
						train_label = dSPlitDict[sent_id]
						if train_label ==1:
							contains_in_train = True
						elif train_label==2:
							contains_in_test = True
						elif train_label==3:
							contains_in_dev = True 

					if phrase==sentence:
						is_a_sentence = True 
					
				#  all neutrals are considered as part of training   
				if sentiment_val >0.4 and sentiment_val<=0.6:
					metadata = "SPLIT:%s"%('unsup')
					istrain='MAYBE'
				elif contains_in_test==True and contains_in_train==False and\
					contains_in_dev==False and is_a_sentence==True:
					metadata = "SPLIT:%s"%('test')
					istrain ="NO"				
				elif contains_in_train ==True and contains_in_test==False and\
					contains_in_dev == False:
					metadata = "SPLIT:%s"%('train')
					istrain='YES'
				else:
					metadata = "SPLIT:%s"%('unsup')
					istrain='MAYBE'
					topic, category ='unsup', 'unsup'

				self.postgres_recorder.insertIntoDocTable(phrase_id, "", \
									phrase, "", metadata) 
				self.postgres_recorder.insertIntoDocTopTable(phrase_id, \
									[topic], [category])
				self._recordParagraphAndSentence(phrase_id, phrase,\
					self.postgres_recorder, topic, istrain)
	
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
		
		#n2vBaseline.runTheBaseline(latent_space_size)

		#Logger.logr.info("Starting Running Iterative Update Method")
		#iterUdateBaseline = IterativeUpdateRetrofitRunner(self.dbstring)
		#iterUdateBaseline.prepareData()
		#iterUdateBaseline.runTheBaseline()
		
		#docBaseLine = Paragraph2VecRunner(self.dbstring)
		#docBaseLine.prepareData()
		#docBaseLine.runTheBaseline(latent_space_size)
		#docBaseLine.runEvaluationTask()
		#docBaseLine.runClassificationTask()

		docBaseLineCEXE = Paragraph2VecCEXERunner(self.dbstring)
		docBaseLineCEXE.prepareData()
		docBaseLineCEXE.runTheBaseline(latent_space_size)
		docBaseLineCEXE.runEvaluationTask()
		docBaseLineCEXE.runClassificationTask()
