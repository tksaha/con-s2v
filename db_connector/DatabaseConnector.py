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
		self.connector = None
		self.dbstring = dbstring 
		self.database, self.username, self.passwd, self.host, self.port = dbstring.split(",")
	
	@abstractmethod
	def	connect_database(dbstring): 
		pass 

	@abstractmethod
	def	disconnect_database(connection):	
		pass 

	def select(fields = [], table = '', where_fields = [], where_values = [], limit1 = '', limit2 = ''):
		try:
			cursor = connector.cursor()
			where_fields_values = [where_field+' = '+str(where_value) for (where_field, where_value) in zip(where_fields, where_values)]
			statement = "SELECT "+', '.join(fields)
			if(table != ''):
				statement += " FROM "+table
			if(where_fields_values != []):
				statement += " WHERE "+' AND '.join(where_fields_values)
			if(limit1 != ''):
				statement += " LIMIT "+limit1+", "+limit2
			print(statement)
			cursor.execute(statement)
			if(cursor.with_rows):
				result = cursor.fetchall()
			else:
				return []
		finally:
			return result
		
	def update(fields = [], values = [], where_fields = [], where_values = [], table = ''):
		try:
			cursor = connector.cursor()
			fields_values = [field+' = '+str(value) for (field, value) in zip(fields, values)]
			where_fields_values = [where_field+' = '+str(where_value) for (where_field, where_value) in zip(where_fields, where_values)]
			statement = "UPDATE "+table+" SET "+ ', '.join(fields_values)+" WHERE "+' AND '.join(where_fields_values)
			print(statement)
			cursor.execute(statement)
		finally:
			connector.commit()

	def insert(values = [], table = '', fields = []):
		try:
			cursor = connector.cursor()
			statement = "INSERT IGNORE INTO "+table
			if(fields != []):
				statement += " ("+', '.join(fields)+") "
			statement += " VALUES ("+', '.join(values)+")"			
			print(statement)
			cursor.execute(statement)
		except mysql.connector.errors.IntegrityError as IntEr:
			log_file = open('errors', 'a')
			log_file.write('\n\nIntEr: '+statement+'\n\n')
		finally:
			connector.commit()		