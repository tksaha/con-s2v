#!/usr/bin/env python
# -*- coding: utf-8 -*-

class BaselineRunner:
	def __init__(self, dbstring):
		"""
		"""
		self.dbstring = dbstring
		self.postgresConnection = PostgresPythonConnector(dbstring)
	

	@abstractmethod
	def prepareData(self):
		"""
		"""
		pass

	@abstractmethod
	def runTheBaseline(self):
		"""
		"""
		pass

	@abstractmethod
	def runEvaluationTask(self):
		"""
		"""
		pass

	@abstractmethod
	def prepareStatisticsAndWrite(self):
		"""
		"""
		pass

