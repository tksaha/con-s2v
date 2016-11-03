#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 
from gensim.models import Word2Vec
from log_manager.log_config import Logger 
from baselineRunner.BaselineRunner import BaselineRunner
import pickle
import math 
import operator 
import multiprocessing 
import subprocess 
import numpy as np 
from word2vec.WordDoc2Vec import WordDoc2Vec
from summaryGenerator.SummaryGenerator import SummaryGenerator


label_sent = lambda id_: 'SENT_%s' %(id_)


class JointLearningSen2VecRunner(BaselineRunner): 

	def __init__(self, *args, **kwargs):
		BaselineRunner.__init__(self, *args, **kwargs)
		self.regsen2vReprFile = os.environ["JOINTS2VRPRFILE"]
		self.dataDir = os.environ['TRTESTFOLDER']
		self.sentsFile = os.environ['P2VCEXECSENTFILE']
		self.num_walks = int(os.environ["NUM_WALKS"])
		self.walk_length = int(os.environ["WALK_LENGTH"])
		self.jointregBetaUNW = float(os.environ['JOINT_BETA'])

		self.Graph = nx.Graph()
		self.cores = multiprocessing.cpu_count()
		self.graphFile = os.environ["GRAPHFILE"]
		self.latReprName = "joint_s2v"

		self.postgresConnection.connectDatabase()
	

	def prepareData(self, pd):
		"""
		It will prepare neighbor data for joint learning.
		Sample neighbor file will look like: 
		2 2 4
		SENT_1  SENT_2 SENT_3 SENT_4 SENT_5
		SENT_1  SENT_3 SENT_4  -1 -1
		
		The very first line gives information about number of 
		lines to read, number of walks and walk_length. 

		If a particular node does not have any neighbor in the 
		walk, then it will have -1 as neighbor which will indicate no 
		neighbor.
		"""
		if pd <= 0: return 0 
		self.Graph = nx.read_gpickle(self.graphFile)
		walkinputFile = open(os.path.join(self.dataDir, "node2vecwalk.txt"))
		joint_nbr_file  = open(os.path.join(self.dataDir,"%s_nbr"%(self.latReprName)), "w")

		line_count = 0 
		for line in walkinputFile: 
			line_count = line_count + 1 

		joint_nbr_file.write("%s %s %s"%(str(line_count),str(self.num_walks),str(self.walk_length)))

		walkinputFile = open(os.path.join(self.dataDir, "node2vecwalk.txt")) # reset position 
		for line in walkinputFile:
			line_elems = line.strip().split(" ")

			for pos in range(0, self.walk_length):
				if pos >= len(line_elems):
					joint_nbr_file.write("-1 ")
				else:
					joint_nbr_file.write("%s "%line_elems[pos])

			joint_nbr_file.write(os.linesep)
			joint_nbr_file.flush()
		joint_nbr_file.close()

	def __dumpVecs(self, reprFile, vecFile, vecRawFile):

		

	def runTheBaseline(self, rbase, latent_space_size):
		"""
		Should we also optimize for window size?
		"""
		if rbase <= 0: return 0 

		self.Graph = nx.read_gpickle(self.graphFile)

		wordDoc2Vec = WordDoc2Vec()
		wPDict = wordDoc2Vec.buildWordDoc2VecParamDict()
		wPDict["cbow"] = str(0) 
		wPDict["sentence-vectors"] = str(1)
		wPDict["min-count"] = str(0)
		wPDict["train"] = "%s.txt"%self.sentsFile
		
		wPDict["size"]= str(latent_space_size)
		args = []


	def generateSummary(self, gs, methodId, filePrefix,\
		 lambda_val=1.0, diversity=False):
		if gs <= 0: return 0
		
		

	def __runEval(self, summaryMethodID, vecFileName, reprName):
		

	def runEvaluationTask(self):
		summaryMethodID = 2 
		
		
	def doHouseKeeping(self):
		"""
		Here, we destroy the database connection.
		"""
		self.postgresConnection.disconnectDatabase()