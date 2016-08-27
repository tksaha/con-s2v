import os 
import sys 
from abc import ABCMeta, abstractmethod



class DocumentReader:
	"""DocumentReader Base"""
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def readDocument():
		pass

	# Protected Methods
	def _splitIntoParagraph(document):
		pass

	def _splitIntoSentence(paragraph):
		pass

	def _splitIntoWords(sentence):
		pass