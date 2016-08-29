# -*- coding: utf-8 -*-
from db_connector.postgresPythonConnector import PostgresPythonConnector as Connector


class PostgresDataRecorder(DataRecorder): 
	"""
	PostgresDataRecorder
	"""
	
	def __init__(self, *args, **kwargs):
		connector = Connector("reuters,naeemul,naeemul,localhost,5432")

	def insertIntoDocTable(self):
		pass
	def insertIntoParTable(self):
		pass
	def insertIntoSenTable(self):
		pass
	def insertIntoDoc_ParTable(self):
		pass
