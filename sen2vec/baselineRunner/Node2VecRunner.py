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
import multiprocessing
import joblib
from joblib import Parallel, delayed



class Node2VecRunner(BaselineRunner): 

	def __init__(self, *args, **kwargs):
		BaselineRunner.__init__(self, *args, **kwargs)
		self.p2vReprFile = os.environ["P2VECSENTRUNNEROUTFILE"]
		self.n2vReprFile = os.environ["N2VOUTFILE"]
		self.interThr = float(os.environ["GINTERTHR"])
		self.intraThr = float(os.environ["GINTRATHR"])
		self.Graph = nx.Graph()
		self.p2vModel = kwargs['p2vmodel'] 
		self.cores = multiprocessing.cpu_count()


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
					if node_id in sentence_id_list: 
						if sim >= self.intraThr:
							self.Graph.add_edge(sentence_id, node_id, weight=sim)
							Logger.logr.info("Adding intra edge (%d, %d) with sim=%f" %(sentence_id, node_id, sim))
						
					else:
						if sim >= self.interThr:
							self.Graph.add_edge(sentence_id, node_id, weight=sim)
							Logger.logr.info("Adding inter edge (%d, %d) with sim=%f" %(sentence_id, node_id, sim))

		Logger.logr.info('The graph is connected  = %d' %(nx.is_connected(self.Graph)))

	def _iterateOverSentences(self, paragraph_id, sentence_id_list):

		
		for sent_result in self.postgresConnection.memoryEfficientSelect(["sentence_id"],\
			["paragraph_sentence"], [["paragraph_id","=",paragraph_id]], \
			[], ["position"]):
			for row_id in range(0,len(sent_result)):
				sentence_id_list.append(sent_result[row_id][0])

		return sentence_id_list
		

	def _iterateOverParagraphs(self, doc_id):

		sentence_id_list = []
		for para_result in self.postgresConnection.memoryEfficientSelect(["paragraph_id"],\
			["document_paragraph"], [["document_id","=",doc_id]], \
			[], ["position"]):
			for row_id in range(0, len(para_result)):
				sentence_id_list = self._iterateOverSentences(\
					para_result[row_id][0], sentence_id_list)

		self._insertGraphEdges(sentence_id_list)


	def prepareData(self):
		"""
		Loops over documents, then paragraphs, and finally over 
		sentences. select(self, fields = [], tables = [], where = [], 
		groupby = [], orderby = [])
		"""
		self.postgresConnection.connect_database()
		self._insertAllNodes()


		for doc_result in self.postgresConnection.memoryEfficientSelect(["id","metadata"],\
			["document"], [], [], ["id"]):
			for row_id in range(0,len(doc_result)):
				self._iterateOverParagraphs(doc_result[row_id][0])
					
		nx.write_gpickle(self.Graph, "Graph_%f_%f.gpickle" %(self.intraThr, self.interThr))
		self.postgresConnection.disconnect_database()


	def runTheBaseline(self, latent_space_size):
		"""
		self.dimension = kwargs['dimension'] 
		self.window_size = kwargs['window_size']
		args.cpu_count = kwargs['cpu_count']
		self.outputfile = kwargs['outputfile']
		self.num_walks = kwargs['num_walks']
		self.walk_length = kwargs['walk_length']
		self.p = kwargs['p']
		self.q = kwargs['q']
		"""

		from node2vec import Node2Vec 
		node2vecInstance = Node2Vec (dimension=latent_space_size, window_size=8,\
			 cpu_count=self.cores, outputfile=self.n2vReprFile,\
			 num_walks=10, walk_length=80, p=4, q=1)
		n2vec = node2vecInstance.get_representation(self.Graph)
	
	def runEvaluationTask(self):
		"""
		"""
		

	def prepareStatisticsAndWrite(self):
		"""
		"""
		
