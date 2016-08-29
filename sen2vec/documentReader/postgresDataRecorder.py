# -*- coding: utf-8 -*-

import os
import sys 

from documentReader.DataRecorder import DataRecorder
from db_connector.postgresPythonConnector import PostgresPythonConnector 


class PostgresDataRecorder(DataRecorder): 
	"""
	PostgresDataRecorder: Records data in postgres tables 
	"""
	
	def __init__(self, *args, **kwargs):
		self.dbstring = "reuters,naeemul,naeemul,localhost,5432"
		self.postgres_connector = PostgresPythonConnector(self.dbstring) 
		self.postgres_connector.connect_database()

	def insertIntoDocTable(self):
		pass
	def insertIntoParTable(self):
		pass
	def insertIntoSenTable(self):
		pass
	def insertIntoDoc_ParTable(self):
		pass
