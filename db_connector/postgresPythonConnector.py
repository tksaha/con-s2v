# -*- coding: utf-8 -*-

import os
import sys
import psycopg2
from DatabaseConnector import * 


class PostgresPythonConnector(DatabaseConnector):
	"""
	PostgresPythonConnector: This class implements the Database connector for Python 

	"""
	def __init__(self, *args, **kwargs):
		DatabaseConnector.__init__(self, *args, **kwargs)

	def	connect_database(self): 
		connection = psycopg2.connect(database=self.database, user=self.username,\
			password=self.passwd, host=self.host, port=self.port)
		print ("Opened database successfully")
		return connection
		
	def	disconnect_database(self,connection):	
		connection.close(); 
		print ("Closed Database successfully")