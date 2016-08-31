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

	def insertIntoTopTable(self, names=[], categories=[]):
		for name, category in zip(names, categories):
			self.postgres_connector.insert([name, category], "topic", ["name", "category"])

	def insertIntoDocTable(self, id, title, content, filename, metadata):
		self.postgres_connector.insert([id, title, content, filename, metadata], "document", ["id", "title", "content", "filename", "metadata"])
		
	def insertIntoParTable(self, content):
		return self.postgres_connector.insert([content], "paragraph", ["content"], 'id')
		
	def insertIntoSenTable(self, content):
		return self.postgres_connector.insert([content], "sentence", ["content"], 'id')

	def insertIntoDoc_TopTable(self, id, names, categories):
		for topic, category in zip(names, categories):
			result = self.postgres_connector.select(["id"], ["topic"], [["name", "=", topic], ["category", "=", category]], [], [])
			if len(result) != 0:
				self.postgres_connector.insert([id, result[0][0]], "document_topic", ["document_id", "topic_id"])

	def insertIntoDoc_ParTable(self, document_id, paragraph_id, position):
		self.postgres_connector.insert([document_id, paragraph_id, position], "document_paragraph", ["document_id", "paragraph_id", "position"])
		
	def insertIntoPar_SenTable(self, paragraph_id, sentence_id, position):
		self.postgres_connector.insert([paragraph_id, sentence_id, position], "paragraph_sentence", ["paragraph_id", "sentence_id", "position"])
