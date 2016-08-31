# -*- coding: utf-8 -*-

import os
import sys 
import logging 

from documentReader.DataRecorder import DataRecorder
from db_connector.PostgresPythonConnector import PostgresPythonConnector 


class PostgresDataRecorder(DataRecorder): 
	"""
	PostgresDataRecorder: Records data in postgres tables 
	"""
	
	def __init__(self, *args, **kwargs):
		"""
		"""
		DataRecorder.__init__(self, *args, **kwargs)
		self.postgres_connector = PostgresPythonConnector(self.dbstring) 
		self.postgres_connector.connect_database()

	def insertIntoTopTable(self, texts=[], categories=[]):
		for topic, category in zip(texts, categories):
			self.postgres_connector.insert([topic, category], "topic", ["text", "category"])

	def insertIntoDocTable(self, id, title, text, file, metadata):
		self.postgres_connector.insert([id, title, text, file, metadata], "document", ["id", "title", "text", "file", "metadata"])
		
	def insertIntoParTable(self):
		"""
		"""
		pass
	def insertIntoSenTable(self):
		"""
		"""
		pass

	def insertIntoDoc_TopTable(self, id, texts, categories):
		for topic, category in zip(texts, categories):
			result = self.postgres_connector.select(["id"], ["topic"], [["text", "=", topic], ["category", "=", category]], [], [])
			self.postgres_connector.insert([id, result[0][0]], "document_topic", ["document_id", "topic_id"])

	def insertIntoDoc_ParTable(self):
		"""
		"""
		pass
