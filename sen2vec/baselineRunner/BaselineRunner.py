#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod
from db_connector.PostgresPythonConnector import PostgresPythonConnector


class BaselineRunner:
	def __init__(self, dbstring, **kwargs):
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

