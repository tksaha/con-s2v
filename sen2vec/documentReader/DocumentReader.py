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
		"""
		return folder.startswith('.') #linux-osx