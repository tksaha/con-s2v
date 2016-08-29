# -*- coding: utf-8 -*-

import os
import sys
import mysql.connector
from DatabaseConnector import * 


class MySQLPythonConnector(DatabaseConnector):
	"""
	MySQLPythonConnector: This class implements the MySQL 
	Database connector for Python 

	"""
	def __init__(self, *args, **kwargs):
		DatabaseConnector.__init__(self, *args, **kwargs)

	def	connect_database(self): 
		self.connector = mysql.connector.connect(database=self.database, user=self.username,\
			password=self.passwd, host=self.host, port=self.port)

		print ("Opened MySQL Database successfully")
		#return connection
		
	def	disconnect_database(self, connection):	
		self.connector.close(); 
		self.connector = None
		print ("Closed MySQL Database successfully")