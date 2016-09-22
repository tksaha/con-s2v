#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import mysql.connector
import logging 
from DatabaseConnector import * 
from log_manager.log_conf import Logger

class MySQLPythonConnector(DatabaseConnector):
	"""
	MySQLPythonConnector: This class implements the MySQL 
	Database connector for Python 

	"""
	def __init__(self, *args, **kwargs):
		"""
		"""
		DatabaseConnector.__init__(self, *args, **kwargs)

	def	connectDatabase(self): 
		"""
		"""
		self.connector = mysql.connector.connect(database=self.database, user=self.username,\
			password=self.passwd, host=self.host, port=self.port)

		logging.info("Opened MySQL Database successfully")

		
	def	disconnectDatabase(self, connection):	
		"""
		"""
		self.connector.close(); 
		self.connector = None
		logging.info("Closed MySQL Database successfully")