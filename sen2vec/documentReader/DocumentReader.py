import os 
import sys 
from abc import ABCMeta, abstractmethod
import logging 



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
	def _splitIntoParagraph(self, document):
		pass

	def _splitIntoSentence(self, paragraph):
		pass

	def _splitIntoWords(self, sentence):
		pass

	def _folder_is_hidden(self, folder):
		return folder.startswith('.') #linux-osx