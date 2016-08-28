# -*- coding: utf-8 -*-
import os 
import sys 
from abc import ABCMeta, abstractmethod

class DataRecorder: 
	"""
	Records data in structured format. See Table schemas 
	inside db_connector folder. [somefile.sql]
	"""
	def __init___(self): 
		pass 

	@abstractmethod
	def insertIntoDocTable(self):
		pass

	@abstractmethod
	def insertIntoParTable(self):
		pass

	@abstractmethod
	def insertIntoSenTable(self):
		pass

	@abstractmethod
	def insertIntoDoc_ParTable(self):
		pass

	# Put Interface for other Tables 
