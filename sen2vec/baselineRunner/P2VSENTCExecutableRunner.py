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
from sklearn import linear_model
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
		self.cores = multiprocessing.cpu_count()
		self.postgresConnection.connectDatabase()
		self.sent2vecMIKOLOVExecutableDir= os.environ['DOC2VECEXECDIR']
		self.utFunction = Utility("Text Utility")
	
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

		args = [self.sent2vecMIKOLOVExecutableDir, "-train","%s.txt"%self.docsFile,\
		    "-output",self.doc2vecOut,\
			"-cbow",str(1),"-size", str(latent_space_size), "-window",str(10),\
			"-negative",str(5),"-hs",str(0),"-sample",str(1e-4) ,\
			"-threads",str(self.cores*2),\
			"-binary",str(0), "-iter",str(20),"-min-count",str(1),\
			"-sentence-vectors", str(1)]

		Logger.logr.info(args)
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = proc.communicate()
		sent2vecModel = Doc2Vec.load_word2vec_format(self.doc2vecOut, binary=False)

		
		args = [self.sent2vecMIKOLOVExecutableDir, "-train","%s.txt"%self.docsFile,\
		    "-output","%s_DBOW"%(self.doc2vecOut),\
			"-cbow",str(0),"-size", str(latent_space_size), "-window",str(10),\
			"-negative",str(5),"-hs",str(0),"-sample",str(1e-4) ,\
			"-threads",str(self.cores*2),\
			"-binary",str(0), "-iter",str(20),"-min-count",str(1),\
			"-sentence-vectors", str(1)]

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
		

	def writeClassificationData(self,result, fileToWrite, s2vDict):
		for row_id in range(0, len(result)):
	 		id_ = result[row_id][0]
	 		topic = result[row_id][1]

	 		vec = s2vDict[id_] 
	 		vec_str = ','.join(str(x) for x in vec)
	 		fileToWrite.write("%s,%s,%s%s"%(id_,vec_str,topic, os.linesep))


	def runEvaluationTask(self):
		"""
		Generate Summary sentences for each document. 
		Write sentence id and corresponding metadata 
		into a file. 
		We should put isTrain=Maybe for the instances which 
		we do not want to incorporate in training and testing. 
		For example. validation set or unsup set
		"""
		method_list = [1, 2]
		sent2vecFile = open("%s.p"%(self.sentReprFile),"rb")
		s2vDict = pickle.load (sent2vecFile)

		for method_id in method_list:
			trainFileToWrite = open("%ss2vtrain_%i.csv"%(self.trainTestFolder, method_id), "w")
			testFileToWrite = open("%ss2vtest_%i.csv"%(self.trainTestFolder, method_id), "w")

			for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", method_id], ['sentence.istrain','=',"'YES'"],\
			 	["sentence.topic","<>","'unsup'"] ], [], []):
				self.writeClassificationData (result, trainFileToWrite, s2vDict)

			for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", method_id], ['sentence.istrain','=',"'NO'"] ], [], []):
			 	self.writeClassificationData (result, testFileToWrite, s2vDict)
			 	
	
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
		method_list = [1, 2]
		
		for method_id in method_list:
			train = pd.read_csv("%ss2vtrain_%i.csv"%(self.trainTestFolder, method_id), header=None)
			test = pd.read_csv("%ss2vtest_%i.csv"%(self.trainTestFolder, method_id), header=None)
									
			train_X, train_Y = self.getXY(train)
			test_X, test_Y = self.getXY(test)
			
			logistic = linear_model.LogisticRegression()
			logit = logistic.fit(train_X, train_Y)
			
			result = pd.DataFrame()
			result['predicted_values'] = logit.predict(test_X)
			result['true_values'] = test_Y
			result.to_csv("%ss2vresult_%i.csv"%(self.trainTestFolder, method_id), index=False)
			
			labels = set(result['true_values'])
			class_labels = {}
			for i, label in enumerate(labels):
				class_labels[label] = label
				
			evaluaiton = ClassificationEvaluation(result['true_values'], result['predicted_values'], class_labels)
			
			evaluationResultFile = open("%ss2veval_%i.txt"%(self.trainTestFolder, method_id), "w")
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
