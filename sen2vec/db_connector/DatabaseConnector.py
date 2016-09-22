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
	def	connectDatabase(dbstring): 
		pass 

	@abstractmethod
	def	disconnectDatabase(connection):	
		pass 
	

	def notZero (self, value):
		return not(value==0)

	def memoryEfficientSelect(self, fields = [], tables = [], where = [], 
		groupby = [], orderby = []):
		"""
		It repeatatively sends results instead of overloading everything 
		in one go. It sends at most 5000 results at a time. If you are anding 
		all clauses in where then you need to append in the list all the 
		operators specifically. [ x or y and z and p]
		"""
		try:
			result = []
			cursor = self.connector.cursor()
			statement = "SELECT %s " %(', '.join(fields))
			statement = "%s FROM %s " %(statement,','.join(tables))
			values = []
			where_clause = []

			if self.notZero(len(where)):
				allAnds = True 
				for wheres in where:
					if len(wheres)==1:
					   where_clause.append("%s" %wheres[0])
					   allAnds = False
					else:
						l, o, r = wheres
						where_clause.append("%s %s %s" %(l,o,r))
				if 	allAnds:
					statement = "%s WHERE %s"%(statement, ' AND '.join(where_clause))
				else:
					statement = "%s WHERE %s "%(statement, ' '. join(where_clause))

			if self.notZero(len(groupby)):
				statement = "%s GROUP BY %s " %(statement, ', '.join(groupby))
			if self.notZero(len(orderby)):
				statement = "%s ORDER BY %s "  %(statement, ', '.join(orderby))

			cursor.execute(statement)
			while True:				
				result = cursor.fetchmany(5000)
				if len(result) > 0:
					yield result 
				else: 
					break 
		except Exception as e:		
			Logger.logr.error("%s%s%s" %(e,": ", statement))
			self.connector.rollback()
			sys.exit()
	
	def insert(self, values = [], table = '', 
		fields = [], returning = ''):
		"""
		"""
		try:
			result = ''
			cursor = self.connector.cursor()
			statement = "INSERT INTO %s" %(table)

			if(fields != []):
				statement = "%s ( %s ) " %(statement, ', '.join(fields))

			placeholder = ["%s"]*len(values)
			statement = "%s VALUES ( %s ) " %(statement, ', '.join(placeholder))

			if(returning != ''):
				statement = "%s RETURNING %s "%(statement, returning)

			cursor.execute(statement, values)
			if returning != '':
				result = cursor.fetchone()[0]
			self.connector.commit()
		except Exception as e:
			Logger.logr.error("%s%s%s" %(e,": ", statement))
			self.connector.rollback()
			sys.exit()
		finally:
			return result

	def executeQuery(self, query):
		try: 
			cursor = self.connector.cursor()
			cursor.execute(query)
			self.connector.commit()
		except Exception as e:
			Logger.logr.info(str(e))
			self.connector.rollback()
			sys.exit()
			

	def truncateTables(self, tables=[]):
		try:
			statement = "truncate %s" % (",".join(tables))
			cursor = self.connector.cursor()
			cursor.execute(statement)
			self.connector.commit()

		except Exception as e:
			Logger.logr.info(str(e))
			self.connector.rollback()
			sys.exit()

