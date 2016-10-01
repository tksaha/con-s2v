#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 
from gensim.models import Word2Vec
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



class Node2VecRunner(BaselineRunner): 

	def __init__(self, *args, **kwargs):
		BaselineRunner.__init__(self, *args, **kwargs)
		self.n2vReprFile = os.environ["N2VOUTFILE"]
		self.p2vReprFile = os.environ["P2VCEXECOUTFILE"]
		self.interThr = float(os.environ["GINTERTHR"])
		self.intraThr = float(os.environ["GINTRATHR"])
		self.dataDir = os.environ['TRTESTFOLDER']
		self.Graph = nx.Graph()
		self.cores = multiprocessing.cpu_count()
		self.graphFile = os.environ["GRAPHFILE"]
		self.latReprName = "n2vecsent"
		self.postgresConnection.connectDatabase()
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

	def getDocID(self, sentence_id):
		for result in self.postgresConnection.memoryEfficientSelect(['doc_id'],\
			['sentence'],[['id','=',str(sentence_id)]],[],[]):
			return int(result[0][0])

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
							

	def prepareData(self, pd):
		"""
		Loops over documents, then paragraphs, and finally over 
		sentences. select(self, fields = [], tables = [], where = [], 
		groupby = [], orderby = [])
		"""
		if pd <= 0: return 0
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
				Logger.logr.info("Number of sentence in sentence"\
					 "dictionary is %i"%len(self.sentenceDict))
				for sentence_result in self.postgresConnection.memoryEfficientSelect(\
					['id','content'],['sentence'],[["doc_id","=",document_id]],[],[]):
					for inrow_id in range(0, len(sentence_result)):
						sentence_id = int(sentence_result[inrow_id][0])
						sentence = sentence_result[inrow_id][1]
						self.sentenceDict[sentence_id] = sentence
				Logger.logr.info("Number of sentence in sentence"\
					 "dictionary is %i"%len(self.sentenceDict))
				self.insertGraphEdges() 
					
		nx.write_gpickle(self.Graph, self.graphFile)
		Logger.logr.info("Total number of edges=%i"%self.Graph.number_of_edges())
		self.Graph = nx.Graph() # Clear
		

	


	def dumpNode2Vec(self, nx_G, reprFile, node2vecFile):

		n2vModel = Word2Vec.load_word2vec_format(reprFile, binary=False)
		Logger.logr.info("Finished Loading WordDoc2Vec Model")
			
		n2vec_dict = {}
		for nodes in nx_G.nodes():
			vec = n2vModel[str(nodes)]
			#Logger.logr.info("Reading a vector of length %s"%vec.shape)
			n2vec_dict[nodes] = vec /  ( np.linalg.norm(vec) +  1e-6)

		pickle.dump(n2vec_dict, node2vecFile)	



	def runTheBaseline(self, rbase, latent_space_size):
		"""
		This will run for both case: one with initialization 
		and another without initialization from para2vec
		"""

		if rbase <= 0: return 0
		Logger.logr.info("Running Node2vec Internal")

		nx_G = nx.read_gpickle(self.graphFile)
		Logger.logr.info("Working a graph with %i edges"%nx_G.number_of_edges())

		############################# Working with Node2Vec Default ############################
		reprFile = "%s_init"%self.n2vReprFile
		initFile = "%s_raw"%self.p2vReprFile
		walkInputFileName = "%s/node2vecwalk.txt"%(self.dataDir)
		node2vecInstance = Node2Vec (dimension=latent_space_size*2, window_size=10,\
			outputfile=reprFile, num_walks=3, walk_length=200, p=4, q=1)

		node2vecInstance.getWalkFile(nx_G, walkInputFileName)
		node2vecFile = open("%s_init.p"%(self.n2vReprFile),"wb")
		node2vecInstance.learnEmbeddings(walkInputFileName, True, initFile, reprFile)
		self.dumpNode2Vec(nx_G, reprFile, node2vecFile)

		############################# Run Node2vec With Initialization from Sen2vec #############
		node2vecFile = open("%s.p"%(self.n2vReprFile),"wb")
		reprFile = self.n2vReprFile
		node2vecInstance.learnEmbeddings(walkInputFileName, False, "",reprFile )
		self.dumpNode2Vec(nx_G, reprFile, node2vecFile)

	

	def generateSummary(self, gs):
		if gs <= 0: return 0
		node2vecFile = open("%s.p"%(self.n2vReprFile),"rb")
		n2vDict = pickle.load (node2vecFile)
		self.populateSummary(3, n2vDict)
		

	def runEvaluationTask(self):
		"""
		Generate Summary sentences for each document. 
		Write sentence id and corresponding metadata 
		into a file. 
		We should put isTrain=Maybe for the instances which 
		we do not want to incorporate in training and testing. 
		For example. validation set or unsup set
		"""

		node2vecFile = open("%s.p"%(self.n2vReprFile),"rb")
		n2vDict = pickle.load (node2vecFile)
		self.generateData(2, self.latReprName, n2vDict)
		self.runClassificationTask(2, self.latReprName)


		node2vecFile = open("%s_init.p"%(self.n2vReprFile),"rb")
		n2vDict = pickle.load (node2vecFile)
		self.generateData(2, "%s_init"%self.latReprName, n2vDict)
		self.runClassificationTask(2, "%s_init"%self.latReprName)
		

	def doHouseKeeping(self):
		"""
		Here, we destroy the database connection.
		"""
		self.postgresConnection.disconnectDatabase()
	
	