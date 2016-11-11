#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 
from gensim.models import Word2Vec
from log_manager.log_config import Logger 
from gensim.models import Doc2Vec
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


class FastSentVariantRunner(BaselineRunner): 

	def __init__(self, *args, **kwargs):
		BaselineRunner.__init__(self, *args, **kwargs)
		self.fastsentReprFile = os.environ["FASTS2VRPRFILE"]
		self.dataDir = os.environ['TRTESTFOLDER']
		self.sentsFile = os.environ['P2VCEXECSENTFILE']
		self.fastsentbeta= float(os.environ['FSENT_BETA'])
		self.cores = multiprocessing.cpu_count()
		self.latReprName = "fsent_s2v"
		self.cores = multiprocessing.cpu_count()
		self.postgresConnection.connectDatabase()
		self.sentenceList = []

	def insertNeighbors(self, sentenceList, nbr_file):
		for pos in range(0, len(sentenceList)):
			nbr_file.write("%s "%label_sent(sentenceList[pos]))
			if pos -1 >= 0:
				nbr_file.write("%s "%label_sent(sentenceList[pos-1]))
			else:
				nbr_file.write("%s "%("-1"))

			if pos+1 < len(sentenceList):
				nbr_file.write("%s "%label_sent(sentenceList[pos+1]))
			else:
				nbr_file.write("%s "%("-1"))
			nbr_file.write(os.linesep)
	

	def prepareData(self, pd):
		"""
		This function will generate the context based on the 
		google's fast sent idea or idea from Felix Hill's 
		sentence representation paper. 
		"""		
		nbr_file = open(os.path.join(self.dataDir, "%s%s"%(self.latReprName,"_nbr")), "w")

		nSent = 0
		for result in self.postgresConnection.memoryEfficientSelect(["count(*)"],\
			['sentence'], [], [], []):
			nSent = int (result[0][0])

		nbr_file.write("%s 1 2%s"%(str(nSent),os.linesep))

		for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
			["document"], [], [], ["id"]):
			for row_id in range(0,len(doc_result)):
				document_id = doc_result[row_id][0]
				#Logger.logr.info("Working for Document id =%i", doc_result[row_id][0])
				self.sentenceList = []
				for sentence_result in self.postgresConnection.memoryEfficientSelect(\
					['id'],['sentence'],[["doc_id","=",document_id]],[],['id']):
					for inrow_id in range(0, len(sentence_result)):
						sentence_id = int(sentence_result[inrow_id][0])
						self.sentenceList.append(sentence_id)
				self.insertNeighbors(self.sentenceList, nbr_file)


	def convert_to_str(self, vec):
		str_ = ""
		for val in vec: 
			str_ ="%s %0.3f"%(str_,val)
		return str_


	def runTheBaseline(self, rbase, latent_space_size):
		"""
		Should we also optimize for window size?
		"""
		if rbase <= 0: return 0 


		wordDoc2Vec = WordDoc2Vec()
		wPDict = wordDoc2Vec.buildWordDoc2VecParamDict()
		wPDict["cbow"] = str(0) 
		wPDict["output"] = os.path.join(self.dataDir , "%s_raw_DBOW"%self.latReprName)
		wPDict["sentence-vectors"] = str(1)
		wPDict["min-count"] = str(0)
		wPDict["train"] = "%s.txt"%self.sentsFile
		wPDict["beta"] = str(self.fastsentbeta)
		
		wPDict["size"]= str(latent_space_size)
		args = []
		neighborFile = os.path.join(self.dataDir,"%s_nbr"%(self.latReprName))
		wPDict["neighborFile"] = neighborFile
		args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 3)
		self._runProcess(args)
		jointvecModel = Doc2Vec.load_word2vec_format(wPDict["output"], binary=False)


		wPDict["cbow"] = str(1) 
		wPDict["output"] = os.path.join(self.dataDir,"%s_raw_DM"%self.latReprName)
		args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 3)
		self._runProcess(args)
		jointvecModelDM = Doc2Vec.load_word2vec_format(wPDict["output"], binary=False)	

		jointvecFile = open("%s.p"%(self.fastsentReprFile),"wb")
		jointvec_dict = {}

		jointvecFile_raw = open("%s_raw.p"%(self.fastsentReprFile),"wb")
		jointvec_raw_dict = {}

		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			 ["sentence"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]	
				vec1 = jointvecModel[label_sent(id_)]
				vec2 = jointvecModelDM[label_sent(id_)]
				vec = np.hstack((vec1,vec2))
				jointvec_raw_dict[id_] = vec 		
				jointvec_dict[id_] = vec /  ( np.linalg.norm(vec) +  1e-6)
				
		Logger.logr.info("Total Number of Sentences written=%i", len(jointvec_raw_dict))			
		pickle.dump(jointvec_dict, jointvecFile)	
		pickle.dump(jointvec_raw_dict, jointvecFile_raw)	

		jointvecFile_raw.close()	
		jointvecFile.close()

	def generateSummary(self, gs, methodId, filePrefix,\
		 lambda_val=1.0, diversity=False):
		if gs <= 0: return 0
		fastsentvecFile = open("%s.p"%(self.fastsentReprFile),"rb")
		fastsentvDict = pickle.load (fastsentvecFile)

		summGen = SummaryGenerator (diverse_summ=diversity,\
			 postgres_connection = self.postgresConnection,\
			 lambda_val = lambda_val)

		# Need a method id for the joint 
	
	def runEvaluationTask(self):
		summaryMethodID = 2 
		fastsentvecFile_raw = open("%s_raw.p"%(self.fastsentReprFile),"rb")
		fastsentvDict_raw = pickle.load(fastsentvecFile_raw)

		if os.environ['EVAL']=='VALID' and os.environ['VALID_FOR']=='CLASS':
			self._runClassificationValidation(summaryMethodID,"%s_raw"%self.latReprName, fastsentvDict_raw)
		elif os.environ['EVAL']=='VALID' and os.environ['VALID_FOR']=='CLUST':
			self._runClusteringValidation(summaryMethodID,"%s_raw"%self.latReprName, fastsentvDict_raw)
		elif os.environ['EVAL']=='TEST' and os.environ['TEST_FOR']=='CLASS':	
			self._runClassification(summaryMethodID,"%s_raw"%self.latReprName, fastsentvDict_raw)
		else:
			self._runClustering(summaryMethodID,"%s_raw"%self.latReprName, fastsentvDict_raw)
		
	def doHouseKeeping(self):
		"""
		Here, we destroy the database connection.
		"""
		self.postgresConnection.disconnectDatabase()