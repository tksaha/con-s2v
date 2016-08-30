# -*- coding: utf-8 -*-

import os
import sys 
import logging 

from documentReader.DataRecorder import DataRecorder
from db_connector.PostgresPythonConnector import PostgresPythonConnector 


class PostgresDataRecorder(DataRecorder): 
	"""
	PostgresDataRecorder: Records data in postgres tables 
	"""
	
	def __init__(self, *args, **kwargs):
		"""
		"""
		DataRecorder.__init__(self, *args, **kwargs)
		self.postgres_connector = PostgresPythonConnector(self.dbstring) 
		#self.postgres_connector.connect_database()

	def insertIntoDocTable(self):
		"""
		"""
		pass
	def insertIntoParTable(self):
		"""
		"""
		pass
	def insertIntoSenTable(self):
		"""
		"""
		pass

	def insertIntoDoc_ParTable(self):
		"""
		"""
		pass
