#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import networkx as nx 
import operator
import numpy as np 
import math
from log_manager.log_config import Logger 
from summaryGenerator.WordBasedGraphGenerator import WordBasedGraphGenerator
from summaryGenerator.PageRankBasedSummarizer import PageRankBasedSummarizer
from log_manager.log_config import Logger 



class SummaryGenerator: 
	def __init__(self, *args, **kwargs):
		"""
		"""
		self.topNSummary = float(os.environ["TOPNSUMMARY"])
		self.intraThrSum_TFIDF = float(os.environ["GTHRSUMTFIDF"])
		self.intraThrSum_LAT = float(os.environ["GTHRSUMLAT"])
		self.dumpingFactor = float(os.environ["DUMPFACTOR"])
		self.postgresConnection = kwargs['postgres_connection']
		self.diversity = kwargs['diverse_summ']
		self.lambda_value = float(kwargs['lambda_val']) # relative weight between ranker 
														# and cosine similarity
		self.vecDict = {}
		self.sentenceDict = {}

	def __constructSingleDocGraphLat(self):
		"""
		"""
		graph = nx.Graph() 
		sortedSentenceDict = sorted(self.sentenceDict.items(),\
		 key=operator.itemgetter(0), reverse=True) 

		for node_id,value in sortedSentenceDict:
			for in_node_id, value in sortedSentenceDict:
				doc_vec_1 = self.vecDict[node_id]
				doc_vec_2 = self.vecDict[in_node_id]
				sim = np.inner(doc_vec_1, doc_vec_2)
				if 	sim >= self.intraThrSum_LAT: 
					graph.add_edge(node_id, in_node_id, weight=sim)

		return graph

	def __dumpSummmaryToTable(self, doc_id, prSummary, idMap, methodID):
		"""
		"""
		position = 1
		for sumSentID, value  in prSummary.getSummary(self.dumpingFactor,\
			 self.diversity, self.lambda_value):
			if 	methodID == 1:
				sumSentID = idMap [sumSentID]
			if  position > len(self.sentenceDict) or\
				  position > math.ceil(len(self.sentenceDict) * self.topNSummary):
				Logger.logr.info("Dumped %i sentence as summary from %i sentence in total" %(position-1, len(self.sentenceDict)))
				break

			self.postgresConnection.insert ([doc_id, methodID, sumSentID, position], "summary",\
			 ["doc_id", "method_id", "sentence_id", "position"])
			position = position +1 

	def __summarizeAndWriteLatentSpaceBasedSummary(self, doc_id, methodID):
		"""
		insert(self, values = [], table = '', 
		fields = [], returning = '')
		Method id 1, 2 for the word based and paragraph2vec 
		based summarizer.
		"""
		nx_G = self.__constructSingleDocGraphLat()
		prSummary = PageRankBasedSummarizer(nx_G = nx_G)
		self.__dumpSummmaryToTable(doc_id, prSummary, "", methodID)

	def __sumarizeAndWriteTFIDFBasedSummary(self, doc_id, methodID):
		"""
		"""
		wbasedGenerator = WordBasedGraphGenerator(\
			sentDictionary=self.sentenceDict,\
		 	threshold=self.intraThrSum_TFIDF)
		nx_G, idMap = wbasedGenerator.generateGraph()

		prSummary = PageRankBasedSummarizer(nx_G = nx_G)
		self.__dumpSummmaryToTable(doc_id, prSummary, idMap, methodID)

	def recordFirstSentenceBaselineSummary(self, methodID):
		"""
		Recording First Sentence as a Baseline Summary
		First sentence as a baseline summary = 21
		"""
		
		position = 1

		for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["document"], [], [], ["id"]):
			for row_id in range(0,len(doc_result)):
				doc_id = doc_result[row_id][0]
				sentence_id = -1
				for result in self.postgresConnection.memoryEfficientSelect(["id"], ["sentence"], [["doc_id", "=", doc_id]], [], ['id']):
					sentence_id = result[0][0]

				self.postgresConnection.insert ([doc_id, methodID, sentence_id, position], "summary",\
			 ["doc_id", "method_id", "sentence_id", "position"])
	

	def populateSummary(self, methodID, vecDict):
		"""
		Method ID one is traditionally assigned to TF-IDF 
		"""
		self.vecDict = vecDict

		if methodID==21:
			self.recordFirstSentenceBaselineSummary(methodID)
			return 0

		for result in self.postgresConnection.memoryEfficientSelect(\
			['id'],['document'],[],[],[]):
			for row_id in range(0,len(result)):
				self.sentenceDict.clear()
				id_ = result[row_id][0]
				for sentence_result in self.postgresConnection.memoryEfficientSelect(\
					['id','content'],['sentence'],[["doc_id","=",id_]],[],[]):
					for inrow_id in range(0, len(sentence_result)):
						sentence_id = int(sentence_result[inrow_id][0])
						sentence = sentence_result[inrow_id][1]
						self.sentenceDict[sentence_id] = sentence 
				if methodID ==1:
					self.__sumarizeAndWriteTFIDFBasedSummary(id_ , methodID)
				else:
					self.__summarizeAndWriteLatentSpaceBasedSummary(id_, methodID)
