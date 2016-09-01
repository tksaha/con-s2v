#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os 
import sys 
from abc import ABCMeta, abstractmethod
import logging 

class DataRecorder: 
	"""
	Records data in structured format. See Table schemas 
	inside db_connector folder. [somefile.sql]
	"""
	def __init__(self, dbstring): 
		self.dbstring = dbstring

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
	def insertIntoTopTable(self):
		pass
		
	@abstractmethod
	def insertIntoDoc_TopTable(self):
		pass
		
	@abstractmethod
	def insertIntoDoc_ParTable(self):
		pass
		
	@abstractmethod
	def insertIntoPar_SenTable(self):
		pass

	# Put Interface for other Tables 
