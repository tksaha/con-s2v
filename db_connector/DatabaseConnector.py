# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod



class DatabaseConnector:
	"""
	Database Connector: Interface for different types of database connection from 
	python
	"""
	def __init__ (self, dbstring):
		self.dbstring = dbstring 
		self.database, self.username, self.passwd, self.host, self.port = dbstring.split(",")
	
	@abstractmethod
	def	connect_database(dbstring): 
		pass 
	@abstractmethod
	def	disconnect_database(connection):	
		pass 