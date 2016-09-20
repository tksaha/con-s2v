#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger 

import nltk
from nltk.tokenize import sent_tokenize

class DocumentReader:
	"""
	DocumentReader Base
	"""
	__metaclass__ = ABCMeta

	def __init__(self):
		pass 


	@abstractmethod
	def readDocument(self):
		pass


	def __recordParagraphAndSentence(self, document_id, doc_content, recorder):
		"""
		"""
		paragraphs = self._splitIntoParagraphs(doc_content)

		for position, paragraph in enumerate(paragraphs):
			paragraph_id = self.postgres_recorder.insertIntoParTable(paragraph)
			self.recorder.insertIntoDoc_ParTable(document_id, paragraph_id, position)
			
			sentences = self._splitIntoSentences(paragraph)
			for sentence_position, sentence in enumerate(sentences):
				sentence_id = self.recorder.insertIntoSenTable(sentence)
				self.recorder.insertIntoPar_SenTable(paragraph_id, sentence_id,\
					sentence_position)

	"""
	Protected Methods 
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

	def _folder_is_hidden(self, folder):
		"""
		What about other windows os?
		http://stackoverflow.com/questions/7099290/how-to-ignore-hidden-files-using-os-listdir-python
		if os.name == 'nt':
	    import win32api, win32con


		def folder_is_hidden(p):
	    	if os.name== 'nt':
		        attribute = win32api.GetFileAttributes(p)
		        return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
	    	else:
	        	return p.startswith('.') #linux-osx
		
		"""
		return folder.startswith('.') #linux-osx