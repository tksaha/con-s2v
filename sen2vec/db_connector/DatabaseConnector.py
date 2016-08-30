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
	
	def select(self, fields = [], tables = [], where = [], groupby = [], orderby = []):
		try:
			result = []
			cursor = self.connector.cursor()
			statement = "SELECT "+', '.join(fields)+" "
			statement += "FROM "+','.join(tables)+" "
			values = []
			where_clause = []
			if len(where) != 0:
				for l, o, r in where:
					where_clause += [l+" "+o+" "+"%s"]
					values += [r]
				statement += "WHERE "+' AND '.join(where_clause)+" "
			if len(groupby) != 0:
				statement += "GROUP BY "+', '.join(groupby)+" "
			if len(orderby) != 0:
				statement += "ORDER BY "+', '.join(orderby)+" "
			cursor.execute(statement, values)
			result = cursor.fetchall()			
		except Exception as e:		
			log_file = open('errors', 'a')
			log_file.write('\n'+e.pgerror+": "+statement+'\n')
			self.connector.rollback()
		finally:
			return result
	
	def insert(self, values = [], table = '', fields = []):
		try:
			cursor = self.connector.cursor()
			statement = "INSERT INTO "+table
			if(fields != []):
				statement += " ("+', '.join(fields)+") "
			placeholder = ["%s"]*len(values)
			statement += " VALUES ("+', '.join(placeholder)+")"
			cursor.execute(statement, values)
			self.connector.commit()
		except Exception as e:		
			log_file = open('errors', 'a')
			log_file.write('\n'+e.pgerror+": "+statement+'\n')
			self.connector.rollback()
