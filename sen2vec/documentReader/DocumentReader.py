import os 
import sys 
from abc import ABCMeta, abstractmethod



class DocumentReader:
	"""DocumentReader Base"""
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def readDocument(self):
		pass

	# Protected Methods
	def _splitIntoParagraph(self, document):
		pass

	def _splitIntoSentence(self, paragraph):
		pass

	def _splitIntoWords(self, sentence):
		pass

	def _folder_is_hidden(self, folder):
		if os.name == 'nt':
			import win32api, win32con
			attribute = win32api.GetFileAttributes(folder)
			return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
		else:
			return folder.startswith('.') #linux-osx