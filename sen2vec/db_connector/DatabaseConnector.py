#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger 

class DatabaseConnector:
	"""
	Database Connector: Interface for different types of 
	database connection from python
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
	
	def select(self, fields = [], tables = [], where = [], 
		groupby = [], orderby = []):
		"""

		"""
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
			Logger.logr.error("%s%s%s%s%s" %(os.linesep, e.pgerror,
					": ", statement, os.linesep))
			self.connector.rollback()
		finally:
			return result

	def memoryEfficientSelect(self, fields = [], tables = [], where = [], 
		groupby = [], orderby = []):
		"""

		"""
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
			while True:				
				result = cursor.fetchmany(5000)
				if len(result) > 0:
					yield result 
				else: 
					break 
		except Exception as e:		
			Logger.logr.error("%s%s%s%s%s" %(os.linesep, e.pgerror,
					": ", statement, os.linesep))
			self.connector.rollback()

		# finally:
		# 	return result
	
	def insert(self, values = [], table = '', 
		fields = [], returning = ''):
		"""

		"""
		try:
			result = ''
			cursor = self.connector.cursor()
			statement = "INSERT INTO "+table
			if(fields != []):
				statement += " ("+', '.join(fields)+") "
			placeholder = ["%s"]*len(values)
			statement += " VALUES ("+', '.join(placeholder)+")"
			if(returning != ''):
				statement += " RETURNING "+returning
			cursor.execute(statement, values)
			if returning != '':
				result = cursor.fetchone()[0]
			self.connector.commit()
		except Exception as e:
			Logger.logr.error("%s%s%s%s%s" %(os.linesep, e.pgerror,
				": ", statement, os.linesep))
			self.connector.rollback()
		finally:
			return result
	def truncate_tables(self, tables=[]):
		try:
			statement = "truncate %s" % (",".join(tables))
			cursor = self.connector.cursor()
			cursor.execute(statement)
			self.connector.commit()

			cursor.execute("ALTER SEQUENCE Topic_id_seq RESTART WITH 1")
			cursor.execute("ALTER SEQUENCE Paragraph_id_seq RESTART WITH 1")
			cursor.execute("ALTER SEQUENCE sentence_id_seq RESTART WITH 1") 

			self.connector.commit()

		except:
			self.connector.rollback()
