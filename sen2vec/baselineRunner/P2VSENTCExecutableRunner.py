#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import numpy as np
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from log_manager.log_config import Logger 
import multiprocessing
from baselineRunner.BaselineRunner import Paragraph2VecSentenceRunner
from collections import namedtuple
from utility.Utility import Utility
import subprocess 
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn import linear_model
from word2vec.WordDoc2Vec import WordDoc2Vec
from evaluation.classificationevaluaiton.ClassificationEvaluation import ClassificationEvaluation 



label_sent = lambda id_: 'SENT_%s' %(id_)


class P2VSENTCExecutableRunner(BaselineRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)
		self.sentsFile = os.environ['P2VCEXECSENTFILE']
		self.sentReprFile = os.environ['P2VCEXECOUTFILE']
		self.doc2vecOut = os.environ['P2VECSENTDOC2VECOUT']
		self.trainTestFolder = os.environ['TRTESTFOLDER']
		self.postgresConnection.connectDatabase()
		self.utFunction = Utility("Text Utility")
		self.latReprName = "p2vsent"
	
	def prepareData(self):
		"""
		Query Sentence Data. We dump sentences with their sentence 
		ids. Pre-pad sentences with null word symbol if the number 
		of words in a sentence 
		is less than 9.
		"""
		sentfiletoWrite = open("%s.txt"%(self.sentsFile),"w")
		for result in self.postgresConnection.memoryEfficientSelect(["id","content"],\
			 ["sentence"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]
				content = gensim.utils.to_unicode(result[row_id][1].strip())
				content = self.utFunction.normalizeText(content, remove_stopwords=0)

				if len(content) < 9:
					n_nulls = 9 - len(content)
					for n in range(0,n_nulls):
						content.insert(0,"null")
				sentfiletoWrite.write("%s %s%s"%(label_sent(id_),' '.join(content), os.linesep))
			sentfiletoWrite.flush()
		sentfiletoWrite.close()


	def runTheBaseline(self, latent_space_size):
		"""
		We run the para2vec Model and then store sen2vec as pickled 
		dictionaries into the output file. 
		"""
		
		sent2vecFile = open("%s.p"%(self.sentReprFile),"wb")
		sent2vec_dict = {}


		wordDoc2Vec = WordDoc2Vec()
		wPDict = wordDoc2Vec.buildWordDoc2VecParamDict()

		wPDict["cbow"], wPDict["sentence-vectors"],wPDict["min-count"] = 0, 0, 0
		wPDict["train"], wPDict["output"] = self.docsFile, self.doc2vecOut
		wPDict["size"]= 300
		args = wordDoc2Vec.buildArgListforW2V(wPDict)
		Logger.logr.info(args)
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = proc.communicate()
		sent2vecModel = Doc2Vec.load_word2vec_format(self.doc2vecOut, binary=False)

		wPDict["cbow"] = 1
		Logger.logr.info(args)
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = proc.communicate()
		sent2vecModelDBOW = Doc2Vec.load_word2vec_format("%s_DBOW"%self.doc2vecOut, binary=False)
		
		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			 ["sentence"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]	
				vec1 = sent2vecModel[label_doc(id_)]
				vec2 = sent2vecModelDBOW[label_doc(id_)]
				vec = np.hstack((vec1,vec2))
				Logger.logr.info("Reading a vector of length %s"%vec.shape)
				sent2vec_dict[id_] = vec /  ( np.linalg.norm(vec) +  1e-6)

		Logger.logr.info("Total Number of Sentences written=%i", len(sent2vec_dict))			
		pickle.dump(sent2vec_dict, sent2vecFile)			
		sent2vecFile.close()
		

	def runEvaluationTask(self):
		"""
		Generate Summary sentences for each document. 
		Write sentence id and corresponding metadata 
		into a file. 
		We should put isTrain=Maybe for the instances which 
		we do not want to incorporate in training and testing. 
		For example. validation set or unsup set
		"""

		sent2vecFile = open("%s.p"%(self.sentReprFile),"rb")
		s2vDict = pickle.load (sent2vecFile)

		self.generateData(self, 1, self.latReprName, s2vDict)
		self.generateData(self, 2, self.latReprName, s2vDict)

		self.runClassificationTask(1, self.latReprName) 
		self.runClassificationTask(2, self.latReprName)
		

	def doHouseKeeping(self):
		"""
		Here, we destroy the database connection.
		"""
		self.postgresConnection.disconnectDatabase()
