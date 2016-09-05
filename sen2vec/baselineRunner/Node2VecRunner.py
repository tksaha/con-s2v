#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from log_manager.log_config import Logger 
from baselineRunner.BaselineRunner import BaselineRunner
from sklearn.metrics.pairwise import cosine_similarity


class Node2VecRunner(BaselineRunner): 

	def __init__(self, *args, **kwargs):
		BaselineRunner.__init__(self, *args, **kwargs)
		self.p2vReprFile = os.environ["P2VECSENTRUNNEROUTFILE"]
		self.n2vReprFile = os.environ["N2VOUTFILE"]
		self.interThr = float(os.environ["GINTERTHR"])
		self.intraThr = float(os.environ["GINTRATHR"])
		self.Graph = nx.Graph()
		self.p2vModel = kwargs['p2vmodel'] 

	def _insertAllNodes(self):
		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["sentence"], [], [], []):
			for row_id in range(0,len(result)):
				id_ = result [row_id] [0]
				self.Graph.add_node(id_)
		Logger.logr.info ("Inserted %d nodes in the graph"\
			 %(self.Graph.number_of_nodes()))

	def _insertGraphEdges(self, sentence_id_list):
		"""
		Process sentences differently for inter and 
		intra documents. 
		"""
		for sentence_id in sentence_id_list:
			for node_id in self.Graph.nodes():
				if node_id != sentence_id:
					doc_vec_1 = self.p2vModel.docvecs['SENT_%i'%sentence_id]
					doc_vec_2 = self.p2vModel.docvecs['SENT_%i'%node_id]
					
					sim =  cosine_similarity(doc_vec_1.reshape(1,-1), doc_vec_2.reshape(1,-1))
					#Logger.logr.info("sim=%f"%sim) 
					if node_id in sentence_id_list: 
						if sim >= self.intraThr:
							self.Graph.add_edge(sentence_id, node_id, weight=sim)
							Logger.logr.info("Adding intra edge with sim=%f" %sim)
						
					else:
						if sim >= self.interThr:
							self.Graph.add_edge(sentence_id, node_id, weight=sim)
							Logger.logr.info("Adding inter edge with sim=%f" %sim)

		Logger.logr.info('The graph is connected  = %d' %(nx.is_connected(self.Graph)))

	def _iterateOverSentences(self, paragraph_id):
		
		for sent_result in self.postgresConnection.memoryEfficientSelect(["sentence_id"],\
			["paragraph_sentence"], [["paragraph_id","=",paragraph_id]], \
			[], ["position"]):

			sentence_id_list = []
			for row_id in range(0,len(sent_result)):
				sentence_id_list.append(sent_result[row_id][0])

			self._insertGraphEdges(sentence_id_list)

	def _iterateOverParagraphs(self, doc_id):
		for para_result in self.postgresConnection.memoryEfficientSelect(["paragraph_id"],\
			["document_paragraph"], [["document_id","=",doc_id]], \
			[], ["position"]):
			for row_id in range(0, len(para_result)):
				self._iterateOverSentences(para_result[row_id][0])


	def prepareData(self):
		"""
		Loops over documents, then paragraphs, and finally over 
		sentences. select(self, fields = [], tables = [], where = [], 
		groupby = [], orderby = [])
		"""
		self.postgresConnection.connect_database()
		self._insertAllNodes()

		for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["document"], [], [], ["id"]):
			for row_id in range(0,len(doc_result)):
				self._iterateOverParagraphs(doc_result[row_id][0])

		self.postgresConnection.disconnect_database()


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
		