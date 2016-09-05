#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from log_manager.log_config import Logger 
from baselineRunner.BaselineRunner import BaselineRunner


class Node2VecRunner(BaselineRunner): 

	def __init__(self, *args, **kwargs):
		BaselineRunner.__init__(self, *args, **kwargs)
		self.p2vReprFile = "%s.p"%os.environment["P2VECSENTRUNNEROUTFILE"]
		self.n2vReprFile = os.environment["N2VOUTFILE"]
		self.interThr = os.environment["GINTERTHR"]
		self.intraThr = os.environment["GINTRATHR"]
		self.Graph = nx.Graph()
		self.p2vModel = Doc2Vec.load(fname) 

	def _insertAllNodes(self):
		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["sentence"], [], [], []):
			for row_id in range(0,len(result)):
				id_ = result [row_id] [0]
				self.Graph.add_node(id_)
		Logger.logr.info ("Inserted %d nodes in the \
			graph ", %len(self.number_of_nodes()))

	def _insertGraphEdges(self, sentence_id_list):
		"""
		Process sentences differently for inter and 
		intra documents. 
		"""
		for sentence_id in sentence_id_list:
			for node_id in G.nodes():
				if node_id != sentence_id:
					sim = self.p2vModel['SENT_%s' %(sentence_id)]['SENT_%s' %(node_id)]
					if node_id in sentence_id: 
						if sim >= self.intraThr:
							G.add_edge(sentence_id, node_id, weight=sim)
						
					else:
						if sim >= self.interThr:
							G.add_edge(sentence_id, node_id, weight=sim)

		Logger.logr.info('The graph is connected  = %d' %G.is_connected())

	def _iterateOverSentences(self, paragraph_id):
		
		for sent_result in self.postgresConnection.memoryEfficientSelect(["sentence_id"],\
			["paragraph_sentence"], ["paragraph_id=%s" %(doc_result[][0])], \
			[], ["position"]):

			sentence_id_list = []
			for row_id in range(0,len(sent_result)):
				sentence_id_list.push(sent_result[row_id][0])

			_insertGraphEdges(sentence_id_list)

	def _iterateOverParagraphs(self, doc_id):
		for para_result in self.postgresConnection.memoryEfficientSelect(["paragraph_id"],\
			["document_paragraph"], ["document_id=%s" %(doc_result[][0])], \
			[], ["position"]):
			for row_id in range(0, len(para_result)):
				_iterateOverSentences(row_id[row_id][0])


	def prepareData(self):
		"""
		Loops over documents, then paragraphs, and finally over 
		sentences. select(self, fields = [], tables = [], where = [], 
		groupby = [], orderby = [])
		"""
		_insertAllNodes()

		for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["document"], [], [], ["id"]):
			for row_id in range(0,len(doc_result)):
				_iterateOverParagraphs(doc_result[row_id][0])

		self.postgresConnection.close()


	def runTheBaseline(self):
		"""
		"""
		pass 
	
	def runEvaluationTask(self):
		"""
		"""
		

	def prepareStatisticsAndWrite(self):
		"""
		"""
		