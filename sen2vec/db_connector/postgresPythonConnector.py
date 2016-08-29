# -*- coding: utf-8 -*-

import os
import sys
import psycopg2
from db_connector.DatabaseConnector import * 


class PostgresPythonConnector(DatabaseConnector):
	"""
	PostgresPythonConnector: This class implements the Postgres 
	Database connector for Python 

	"""
	def __init__(self, *args, **kwargs):
		DatabaseConnector.__init__(self, *args, **kwargs)

	def	connect_database(self): 
		self.connector = psycopg2.connect(database=self.database, user=self.username,\
			password=self.passwd, host=self.host, port=self.port)

		print ("Opened Postgres Database successfully")
		#return connection
		
	def	disconnect_database(self, connection):	
		self.connector.close(); 
		self.connector = None
		print ("Closed Postgres Database successfully")