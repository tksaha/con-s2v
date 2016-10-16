#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger 

import nltk
from nltk.tokenize import sent_tokenize
from utility.Utility import Utility
import gensim

class DocumentReader:
	"""
	DocumentReader Base
	"""
	__metaclass__ = ABCMeta

	def __init__(self):
		self.utFunction = Utility("Text Utility")


	@abstractmethod
	def readDocument(self):
		pass


	"""
	Protected Methods: Accessed by subclasses 
	"""
	def _splitIntoParagraphs(self, document):
		"""
		This is a rough heuristics. 
		"""
		return document.split("%s%s" %(os.linesep, os.linesep))

	def _splitIntoSentences(self, paragraph):
		"""
		"""
		return sent_tokenize(paragraph)

	def _splitIntoWords(self, sentence):
		pass

	def _folderISHidden(self, folder):
		"""
		http://stackoverflow.com/questions/7099290/how-to-ignore-hidden-files-using-os-listdir-python
		"""
		return folder.startswith('.') #linux-osx


	def _recordParagraphAndSentence(self, document_id, doc_content, recorder, topic, istrain, skip_short=False):
		"""
		It seems Mikolov and others did n't remove the stop words. So, we also do the 
		same for vector construction. 
		"""
		paragraphs = self._splitIntoParagraphs(doc_content)

		for position, paragraph in enumerate(paragraphs):
			paragraph_id = self.postgres_recorder.insertIntoParTable(paragraph)
			recorder.insertIntoDocParTable(document_id, paragraph_id, position)
			
			sentences = self._splitIntoSentences(paragraph)
			for sentence_position, sentence in enumerate(sentences):
				sentence_ = self.utFunction.normalizeText(sentence, 0)
				if len(sentence_) <=2:
					continue 
				elif len(sentence_) < 4 and skip_short == True:
					continue

				sentence = sentence.replace("\x03","")
				sentence = sentence.replace("\x02","")
				sentence = sentence.replace('\n', ' ').replace('\r', '').strip()
				sentence_id = recorder.insertIntoSenTable(sentence,\
					 topic, istrain, document_id, paragraph_id)
				recorder.insertIntoParSenTable(paragraph_id, sentence_id,\
					sentence_position)

	def _getTextFromFile(self, file):
		"""
		http://stackoverflow.com/questions/7409780/reading-entire-file-in-python
		"""	
		with open(file, encoding='utf-8', errors='ignore') as f:
			return f.read()



	def _getTopics(self, rootDir):
		Logger.logr.info("Starting Reading Topic")
		topic_names, categories = [], []
		for dirName, subdirList, fileList in os.walk(rootDir):
			for topics in subdirList:
				topic_names.append(topics)
				categories.append(topics.split('.')[0])
		self.postgres_recorder.insertIntoTopTable(topic_names, categories)				
		Logger.logr.info("[%i] Topic reading complete." %(len(topic_names)))
		return topic_names

	
