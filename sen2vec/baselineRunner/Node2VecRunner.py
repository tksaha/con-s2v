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
		self.p2vReprFile = os.environ["P2VECSENTRUNNEROUTFILE"]
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


	def _insertAllNodes(self):
		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["sentence"], [], [], []):
			for row_id in range(0,len(result)):
				id_ = result [row_id] [0]
				self.Graph.add_node(id_)
		Logger.logr.info ("Inserted %d nodes in the graph"\
			 %(self.Graph.number_of_nodes()))

	def _insertGraphEdges(self):
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
							#Logger.logr.info("Adding intra edge (%d, %d) with sim=%f" %(sentence_id, node_id, sim))
						
					else:
						if sim >= self.interThr:
							self.Graph.add_edge(sentence_id, node_id, weight=sim)
							#Logger.logr.info("Adding inter edge (%d, %d) with sim=%f" %(sentence_id, node_id, sim))

		#Logger.logr.info('The graph is connected  = %d' %(nx.is_connected(self.Graph)))

	def _iterateOverSentences(self, paragraph_id):

		
		for sent_result in self.postgresConnection.memoryEfficientSelect(["sentence_id"],\
			["paragraph_sentence"], [["paragraph_id","=",paragraph_id]], \
			[], ["position"]):
			for row_id in range(0,len(sent_result)):
				self.sentenceDict[sent_result[row_id][0]] = "1"
		
	def _constructSingleDocGraphP2V(self):
		graph = nx.Graph() 
		sortedSentenceDict = sorted(self.sentenceDict.items(), key=operator.itemgetter(0), reverse=True) 

		for node_id,value in sortedSentenceDict:
			for in_node_id, value in sortedSentenceDict:
				doc_vec_1 = self.s2vDict[node_id]
				doc_vec_2 = self.s2vDict[in_node_id]
				sim = np.inner(doc_vec_1, doc_vec_2)
				if 	sim > self.intraThrSummary: 
					graph.add_edge(node_id, in_node_id, weight=sim)

		return graph

	def _dumpSummmaryToTable(self, doc_id, prSummary, idMap, methodID):
		position = 1
		for sumSentID, value  in prSummary.getSummary(self.dumpingFactor):
			if 	methodID == 1:
				sumSentID = idMap [sumSentID]

			self.postgresConnection.insert ([doc_id, methodID, sumSentID, position], "summary",\
			 ["doc_id", "method_id", "sentence_id", "position"])


			if  position > len(self.sentenceDict) or  position > math.ceil(len(self.sentenceDict) * self.topNSummary):
				Logger.logr.info("Dumped %i sentence as summary from %i sentence in total" %(position, len(self.sentenceDict)))
				break
			position = position +1 

	def _summarizeAndWriteLabels(self, doc_id):
		"""
		insert(self, values = [], table = '', 
		fields = [], returning = '')
		"""

		wbasedGenerator = WordBasedGraphGenerator (sentDictionary=self.sentenceDict, threshold=self.intraThrSummary)
		nx_G, idMap = wbasedGenerator.generateGraph()
		prSummary = PageRankBasedSummarizer(nx_G = nx_G)
		self._dumpSummmaryToTable(doc_id, prSummary, idMap, 1)

		nx_G = self._constructSingleDocGraphP2V()
		prSummary = PageRankBasedSummarizer(nx_G = nx_G)
		self._dumpSummmaryToTable(doc_id, prSummary, "", 2)
		


	def _iterateOverParagraphs(self, doc_id):
		"""
		Prepare a large graph. Prepare per document graph, 
		summarize and label as train or test.
		"""
		self.sentenceDict.clear()

		for para_result in self.postgresConnection.memoryEfficientSelect(["paragraph_id"],\
			["document_paragraph"], [["document_id","=",doc_id]], \
			[], ["position"]):
			for row_id in range(0, len(para_result)):
				self._iterateOverSentences(para_result[row_id][0])

		
	
		for id_ in self.sentenceDict.keys():
			for sent_result in self.postgresConnection.memoryEfficientSelect(["id", "content"],\
				["sentence"], [["id", "=", id_]], [], []):
				for row_id in range(len(sent_result)):
					self.sentenceDict[id_] = sent_result[row_id][1]


		self._summarizeAndWriteLabels(doc_id)
		#self._insertGraphEdges()


	def prepareData(self):
		"""
		Loops over documents, then paragraphs, and finally over 
		sentences. select(self, fields = [], tables = [], where = [], 
		groupby = [], orderby = [])
		"""
		self.postgresConnection.connect_database()
		self._insertAllNodes()

		p2vfileToRead = open ("%s.p" %self.p2vReprFile, "rb")
		self.s2vDict = pickle.load(p2vfileToRead)

		for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["document"], [], [], ["id"]):
			for row_id in range(0,len(doc_result)):
				Logger.logr.info("Working for Document id =%i", doc_result[row_id][0])
				self._iterateOverParagraphs(doc_result[row_id][0])
					
		nx.write_gpickle(self.Graph, self.graphFile)
		Logger.logr.info("Total number of edges=%i"%self.Graph.number_of_edges())

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
		Logger.logr.info("Running Node2vec Internal")
		node2vecInstance = Node2Vec (dimension=latent_space_size, window_size=8,\
			 cpu_count=self.cores, outputfile=self.n2vReprFile,\
			 num_walks=10, walk_length=10, p=4, q=1)
		n2vec = node2vecInstance.get_representation(self.Graph)
		return self.Graph
	
	def runEvaluationTask(self):
		"""
		"""
		

	def prepareStatisticsAndWrite(self):
		"""
		"""
		
