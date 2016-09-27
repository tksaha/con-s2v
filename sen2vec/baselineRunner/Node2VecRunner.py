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
import pickle
import math 
import operator 
import multiprocessing 
import numpy as np 
from node2vec.Node2Vec import Node2Vec 
from summaryGenerator.WordBasedGraphGenerator import WordBasedGraphGenerator
from summaryGenerator.PageRankBasedSummarizer import PageRankBasedSummarizer


class Node2VecRunner(BaselineRunner): 

	def __init__(self, *args, **kwargs):
		BaselineRunner.__init__(self, *args, **kwargs)
		self.n2vReprFile = os.environ["N2VOUTFILE"]
		self.interThr = float(os.environ["GINTERTHR"])
		self.intraThr = float(os.environ["GINTRATHR"])
		self.intraThrSummary = float(os.environ["GTHRSUM"])
		self.dumpingFactor = float(os.environ["DUMPFACTOR"])
		self.topNSummary = float(os.environ["TOPNSUMMARY"])
		self.Graph = nx.Graph()
		self.cores = multiprocessing.cpu_count()
		self.graphFile = os.environ["GRAPHFILE"]
		self.s2vDict = {}
		self.sentenceDict = {}


	def insertAllNodes(self):
		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["sentence"], [], [], []):
			for row_id in range(0,len(result)):
				id_ = result [row_id] [0]
				self.Graph.add_node(id_)
		Logger.logr.info ("Inserted %d nodes in the graph"\
			 %(self.Graph.number_of_nodes()))

	def insertGraphEdges(self):
		"""
		Process sentences differently for inter and 
		intra documents. 
		"""
		for sentence_id in self.sentenceDict.keys():
			for node_id in self.Graph.nodes():
				if node_id != sentence_id:	
					doc_vec_1 = self.s2vDict[node_id]
					doc_vec_2 = self.s2vDict[sentence_id]
					sim = np.inner(doc_vec_1, doc_vec_2)
					if node_id in self.sentenceDict.keys(): 
						if sim >= self.intraThr:
							self.Graph.add_edge(sentence_id, node_id, weight=sim)	
					else:
						if sim >= self.interThr:
							self.Graph.add_edge(sentence_id, node_id, weight=sim)
							

	def prepareData(self):
		"""
		Loops over documents, then paragraphs, and finally over 
		sentences. select(self, fields = [], tables = [], where = [], 
		groupby = [], orderby = [])
		"""
		self.postgresConnection.connectDatabase()
		self.insertAllNodes()

		p2vfileToRead = open ("%s.p" %self.p2vReprFile, "rb")
		self.s2vDict = pickle.load(p2vfileToRead)

		for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["document"], [], [], ["id"]):
			for row_id in range(0,len(doc_result)):
				document_id = doc_result[row_id][0]
				Logger.logr.info("Working for Document id =%i", doc_result[row_id][0])
				self.sentenceDict.clear()
				for setence_result in self.postgresConnection.memoryEfficientSelect(\
					['id','content'],['sentence'],[["doc_id","=",document_id]],[],[]):
					for inrow_id in range(0, len(inrow_id)):
						sentence_id = int(sentence_result[inrow_id][0])
						sentence = sentence_result[inrow_id][1]
						self.sentenceDict[sentence_id] = sentence
				self.insertGraphEdges() 
					
		nx.write_gpickle(self.Graph, self.graphFile)
		Logger.logr.info("Total number of edges=%i"%self.Graph.number_of_edges())

		self.postgresConnection.disconnectDatabase()


	def runTheBaseline(self, latent_space_size):
		"""
		"""
		Logger.logr.info("Running Node2vec Internal")
		node2vecInstance = Node2Vec (dimension=latent_space_size, window_size=10,\
			outputfile=self.n2vReprFile, num_walks=10, walk_length=10, p=4, q=1)
		n2vec = node2vecInstance.get_representation(self.Graph)
		return self.Graph
	
	def runEvaluationTask(self):
		"""
		"""
		

	def prepareStatisticsAndWrite(self):
		"""
		"""
		
