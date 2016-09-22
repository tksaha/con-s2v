#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys 

from documentReader.DataRecorder import DataRecorder
from db_connector.PostgresPythonConnector import PostgresPythonConnector 
from log_manager.log_config import Logger 


class PostgresDataRecorder(DataRecorder): 
	"""
	PostgresDataRecorder: Records data in postgres tables 
	"""
	
	def __init__(self, *args, **kwargs):
		"""
		"""
		DataRecorder.__init__(self, *args, **kwargs)
		self.postgres_connector = PostgresPythonConnector(self.dbstring) 
		self.postgres_connector.connectDatabase()

	def trucateTables(self):
		self.postgres_connector.truncateTables(["topic", "document",\
					"document_topic", "paragraph",\
					"document_paragraph", "sentence",\
					"paragraph_sentence", "summary"])

	def alterSequences(self):
		self.postgres_connector.executeQuery("ALTER SEQUENCE paragraph_id_seq RESTART WITH 1")
		self.postgres_connector.executeQuery("ALTER SEQUENCE sentence_id_seq RESTART WITH 1")
		self.postgres_connector.executeQuery("ALTER SEQUENCE \"Topic_id_seq\" RESTART WITH 1")
		Logger.logr.info("Altered the Sequences")

	def insertIntoTopTable(self, names=[], categories=[]):
		"""
		"""
		for name, category in zip(names, categories):
			self.postgres_connector.insert([name, category], "topic", ["name", "category"])

	def insertIntoDocTable(self, id, title, content, filename, metadata):
		"""
		"""
		self.postgres_connector.insert([id, title, content, filename, metadata], "document", ["id", "title", "content", "filename", "metadata"])
		
	def insertIntoParTable(self, content):
		"""
		"""
		return self.postgres_connector.insert([content], "paragraph", ["content"], 'id')
		
	def insertIntoSenTable(self, content, topic, istrain, document_id, paragraph_id):
		"""
		"""
		return self.postgres_connector.insert([content, topic,\
			istrain, document_id, paragraph_id], "sentence", ["content","topic","istrain","doc_id","par_id"], 'id')

	def insertIntoDocTopTable(self, id, names, categories):
		"""
		"""
		for topic, category in zip(names, categories):
			for result in self.postgres_connector.memoryEfficientSelect(["id"], ["topic"], [["name", "=", "'%s'"%topic], ["category", "=", "'%s'"%category]], [], []):
				self.postgres_connector.insert([id, result[0][0]], "document_topic", ["document_id", "topic_id"])

	def insertIntoDocParTable(self, document_id, paragraph_id, position):
		"""
		"""
		self.postgres_connector.insert([document_id, paragraph_id, position], "document_paragraph", ["document_id", "paragraph_id", "position"])
		
	def insertIntoParSenTable(self, paragraph_id, sentence_id, position):
		"""
		"""
		self.postgres_connector.insert([paragraph_id, sentence_id, position], "paragraph_sentence", ["paragraph_id", "sentence_id", "position"])
