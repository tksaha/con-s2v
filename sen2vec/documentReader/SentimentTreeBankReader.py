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
		topic_names = ['pos', 'neg','dev']
		categories = ['pos', 'neg', 'dev']

		self.postgres_recorder.insertIntoTopTable(topic_names, categories)				
		Logger.logr.info("[%i] Topic reading complete." %(len(topic_names)))
		return topic_names

	def readDSplit(fileName):
		"""
		1 Train, 2 Test, 3 dev
		"""
		line_count = 0 
		dSPlitDict = {}
		for line in open(fileName):
			if line_count == 0: 
				continue
			else:
				line_count = line_count + 1
				doc_id,_, splitid = line.strip().partition(" ")

		dSPlitDict[int(doc_id)] = int(splitid)
		Logger.logr.info("Finished reading %i sentences and their splits"%line_count)

		return dSPlitDict;

	def readSentences(fileName):
		line_count = 0
		sentenceDict = {}
		for line in open(fileName):
			if line_count == 0:
				continue
			else:
				line_count = line_count + 1
				doc_id,_,sentence = line.strip().partition(",")
				sentenceDict[int(doc_id)] = sentence.strip()
		Logger.logr.info("Finished reading %i sentence"%line_count)

	def phraseToSentiment(fileName):
		line_count = 0 
		phraseToSentimentDict = {}

		for line in open(fileName):
			if line_count == 0:
				continue
			else:
				line_count = line_count + 1
				phrase_id,_, sentiment = line.strip().split("|")
				phraseToSentimentDict[int(phrase_id)] = float(sentiment)
		Logger.logr.info("Finished reading %i phrases"%line_count)

	def getTopicCategory(sentiment_val):
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
			return ('neu', 'neu')

	def readDocument(self, ld): 
		"""
		SKip neutral phrases 
		"""

		if ld <= 0: return 0 			
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.alterSequences()
		topic_names = self.readTopic()

		allPhrasesFile = "%s/dictionary.txt"%(self.folderPath)
		dSPlitDict = self.readDSplit("%s/dataSplit.txt"%self.folderPath)
		sentenceDict = self.readSentences("%s/dataSentences.txt"%self.folderPath)
		phraseToSentiment = self.phraseToSentiment("%s/sentiment_labels.txt"%self.folderPath)

		for line in open(allPhrasesFile):
				phrase, _ , phrase_id = line.strip().split("|")
				contains_in_train, contains_in_test, contains_in_dev, is_a_sentence = False, False, False, False
				sentiment_val = phraseToSentimentDict[phrase_id]
				if sentiment_val >0.4 and sentiment_val<=0.6:
					continue 
				topic, category = getTopicCategory(sentiment_val)
				for sent_id, sentence in sentenceDict.iteritems():
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
					   

				if contains_in_test==True and contains_in_train==False and\
					contains_in_dev==False and is_a_sentence==True:
					metadata = "SPLIT:%s"%('test')
					istrain ="NO"
					self.postgres_recorder.insertIntoDocTable(phrase_id, "", \
									phrase, "", metadata) 

				elif contains_in_train ==True and contains_in_test==False and\
					contains_in_dev == False:
					metadata = "SPLIT:%s"%('train')
					istrain='YES'
					self.postgres_recorder.insertIntoDocTable(phrase_id, "", \
									phrase, "", metadata)
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
		allPhrasesFile.close()

		Logger.logr.info("Document reading complete.")
		return 1


