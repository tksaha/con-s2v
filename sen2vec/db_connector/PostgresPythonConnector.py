# -*- coding: utf-8 -*-

import os
import sys
import psycopg2
from db_connector.DatabaseConnector import * 

#from log_manager.log_conf import Logger

class PostgresPythonConnector(DatabaseConnector):
	"""
	PostgresPythonConnector: This class implements the Postgres 
	Database connector for Python 

	"""
	def __init__(self, *args, **kwargs):
		"""
		"""
		DatabaseConnector.__init__(self, *args, **kwargs)

	def	connect_database(self): 
		"""
		"""
		try: 
			self.connector = psycopg2.connect(database=self.database, user=self.username,\
			password=self.passwd, host=self.host, port=self.port)
			Logger.logr.info("Opened Postgres Database successfully")
		except:
			Logger.logr.info(" Postgres Database connection unsuccessful")
		
		
	def	disconnect_database(self, connection):	
		"""
		"""
		self.connector.close(); 
		self.connector = None
		logger.info("Closed Postgres Database successfully")