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
from baselineRunner.BaselineRunner import BaselineRunner
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
from utility.Utility import Utility
import pandas as pd
from sklearn import linear_model
from evaluation.classificationevaluaiton.ClassificationEvaluation import ClassificationEvaluation 
import subprocess 

"""
https://docs.python.org/2/library/collections.html
"""
ImdbDocument = namedtuple('Document', 'words tags')
label_doc = lambda id_: 'DOC_%s' %(id_)


class Paragraph2VecCEXERunner(BaselineRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)
		self.docsFile = os.environ['P2VECRUNNERINFILE']
		self.doc2vecOut = os.environ['P2VDOCOUT']
		self.docReprFile = os.environ['P2VECRUNNERCEXEOUTFILE']
		self.trainTestFolder = os.environ['TRTESTFOLDER']
		self.cores = multiprocessing.cpu_count()
		self.postgresConnection.connectDatabase()
		self.doc2vecMIKOLOVExecutableDir= os.environ['DOC2VECEXECDIR']
		self.utFunction = Utility("Text Utility")
	
	def prepareData(self):
		"""
		Query Sentence Data. We dump sentences with their sentence 
		ids. Pre-pad sentences with null word symbol if the number 
		of words in a sentence 
		is less than 9.
		"""
		docfiletoWrite = open("%s.txt"%(self.docsFile),"w")
		for result in self.postgresConnection.memoryEfficientSelect(["id","content"],\
			 ["document"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]
				content = gensim.utils.to_unicode(result[row_id][1].strip())
				content = self.utFunction.normalizeText(content, remove_stopwords=0)

				if len(content) < 9:
					n_nulls = 9 - len(content)
					for n in range(0,n_nulls):
						content.insert(0,"null")
				docfiletoWrite.write("%s %s%s"%(label_doc(id_),' '.join(content), os.linesep))
			docfiletoWrite.flush()

		docfiletoWrite.close()


	def runTheBaseline(self, latent_space_size):
		"""
		We run the para2vec Model and then store sen2vec as pickled 
		dictionaries into the output file. 
		"""
		
		doc2vecFile = open("%s.p"%(self.docReprFile),"wb")
		doc2vec_dict = {}

		args = [self.doc2vecMIKOLOVExecutableDir, "-train","%s.txt"%self.docsFile,\
		    "-output",self.doc2vecOut,\
			"-cbow",str(1),"-size", str(latent_space_size), "-window",str(10),\
			"-negative",str(5),"-hs",str(0),"-sample",str(1e-4) ,\
			"-threads",str(self.cores*2),\
			"-binary",str(0), "-iter",str(20),"-min-count",str(1),\
			"-sentence-vectors", str(1)]

		Logger.logr.info(args)
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = proc.communicate()
		doc2vecModel = Doc2Vec.load_word2vec_format(self.doc2vecOut, binary=False)


		args = [self.doc2vecMIKOLOVExecutableDir, "-train","%s.txt"%self.docsFile,\
		    "-output","%s_DBOW"%(self.doc2vecOut),\
			"-cbow",str(0),"-size", str(latent_space_size), "-window",str(10),\
			"-negative",str(5),"-hs",str(0),"-sample",str(1e-4) ,\
			"-threads",str(self.cores*2),\
			"-binary",str(0), "-iter",str(20),"-min-count",str(1),\
			"-sentence-vectors", str(1)]

		Logger.logr.info(args)
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = proc.communicate()
		doc2vecModelDBOW = Doc2Vec.load_word2vec_format("%s_DBOW"%self.doc2vecOut, binary=False)

		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			 ["document"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]	
				vec1 = doc2vecModel[label_doc(id_)]
				vec2 = doc2vecModelDBOW[label_doc(id_)]
				vec = np.hstack((vec1,vec2))
				Logger.logr.info("Reading a vector of length %s"%vec.shape)
				doc2vec_dict[id_] = vec /  ( np.linalg.norm(vec) +  1e-6)

		Logger.logr.info("Total Number of Documents written=%i", len(doc2vec_dict))			
		pickle.dump(doc2vec_dict, doc2vecFile)			
		doc2vecFile.close()
		

	def writeClassificationData(self,result, fileToWrite, d2vDict, topicIdDict):
		for row_id in range(0, len(result)):
	 		id_ = result[row_id][0]
	 		topic = topicIdDict[int(result[row_id][1])]

	 		vec = d2vDict[id_] 
	 		vec_str = ','.join(str(x) for x in vec)
	 		fileToWrite.write("%s,%s,%s%s"%(id_,vec_str,topic,os.linesep))


	def runEvaluationTask(self):
		"""
		Here from observation: unsup belongs to topic_id 3. 
		"""
	
		doc2vecFile = open("%s.p"%(self.docReprFile),"rb")
		d2vDict = pickle.load (doc2vecFile)

		trainFileToWrite = open("%sd2vtrain.csv"%(self.trainTestFolder), "w")
		testFileToWrite = open("%sd2vtest.csv"%(self.trainTestFolder), "w")

		topicIdDict = {}
		for result in self.postgresConnection.memoryEfficientSelect(["id","name"], \
			["topic"],[],[],[]):
			for row_id in range(0,len(result)):
				topicIdDict[int(result[row_id][0])] = result[row_id][1]


		for result in self.postgresConnection.memoryEfficientSelect(["document.id",\
			"document_topic.topic_id", "metadata"],\
			["document,document_topic"],[["document.id","=", "document_topic.document_id"],\
			["topic_id","<>","3"], ["metadata","like","'SPLIT:train'"]],[],[] ):
				self.writeClassificationData (result, trainFileToWrite, d2vDict, topicIdDict)
		Logger.logr.info("Finished populating Training")

		for result in self.postgresConnection.memoryEfficientSelect(["document.id",\
			"document_topic.topic_id", "metadata"],\
			["document,document_topic"],[["document.id","=", "document_topic.document_id"],\
			["topic_id","<>","3"], ["metadata","like","'SPLIT:test'"]],[],[] ):
				self.writeClassificationData (result, testFileToWrite, d2vDict, topicIdDict)
		
		Logger.logr.info("Finished Populating Test")
		trainFileToWrite.flush()
		testFileToWrite.flush() 


	
	def getXY(self, data):
		"""
		This function assumes that the data (pandas DF) has id in the 
		first column, label in the last column and features 
		in the middle. It returns features as X and label as Y.
		"""
		X = data[data.columns[1: data.shape[1]-1 ]]
		Y = data[data.shape[1]-1]
		return (X, Y)
	
	
	def runClassificationTask(self):
		"""
		This function uses the generated train and test 
		files to build a logistic regression model using 
		scikit-learn. It will save the results 
		into files.
		"""
		
		train = pd.read_csv("%sd2vtrain.csv"%(self.trainTestFolder), header=None)
		test = pd.read_csv("%sd2vtest.csv"%(self.trainTestFolder), header=None)
							
		train_X, train_Y = self.getXY(train)
		test_X, test_Y = self.getXY(test)
		
		logistic = linear_model.LogisticRegression()
		logit = logistic.fit(train_X, train_Y)
		
		result = pd.DataFrame()
		result['predicted_values'] = logit.predict(test_X)
		result['true_values'] = test_Y
		result.to_csv("%sd2vresult.csv"%(self.trainTestFolder), index=False)
		
		labels = set(result['true_values'])
		class_labels = {}
		for i, label in enumerate(labels):
			class_labels[label] = label
			
		evaluaiton = ClassificationEvaluation(result['true_values'], result['predicted_values'], class_labels)
		
		evaluationResultFile = open("%sd2veval.txt"%(self.trainTestFolder), "w")
		evaluationResultFile.write("%s%s%s" %("######Classification Report######\n", \
					evaluaiton._getClassificationReport(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Accuracy Score######\n", \
					evaluaiton._getAccuracyScore(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Confusion Matrix######\n", \
					evaluaiton._getConfusionMatrix(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Cohen's Kappa######\n", \
					evaluaiton._getCohenKappaScore(), "\n\n"))
					
		Logger.logr.info("Evaluation Completed.")


	def prepareStatisticsAndWrite(self):
		"""
		"""
		self.postgresConnection.disconnectDatabase()