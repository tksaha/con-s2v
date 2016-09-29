#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math
from abc import ABCMeta, abstractmethod
import networkx as nx 
import pandas as pd
from log_manager.log_config import Logger
from sklearn.dummy import DummyClassifier
from sklearn import linear_model
import operator
import numpy as np 
from db_connector.PostgresPythonConnector import PostgresPythonConnector
from summaryGenerator.WordBasedGraphGenerator import WordBasedGraphGenerator
from summaryGenerator.PageRankBasedSummarizer import PageRankBasedSummarizer
from evaluation.classificationevaluaiton.ClassificationEvaluation import ClassificationEvaluation 


class BaselineRunner:
	def __init__(self, dbstring, **kwargs):
		"""
		"""
		self.dbstring = dbstring
		self.postgresConnection = PostgresPythonConnector(dbstring)
		self.topNSummary = float(os.environ["TOPNSUMMARY"])
		self.intraThrSummary = float(os.environ["GTHRSUM"])
		self.dumpingFactor = float(os.environ["DUMPFACTOR"])
		self.trainTestFolder = os.environ['TRTESTFOLDER']
		self.vecDict = {}
		self.sentenceDict = {}

	@abstractmethod
	def prepareData(self):
		"""
		"""
		pass

	@abstractmethod
	def runTheBaseline(self):
		"""
		"""
		pass

	@abstractmethod
	def runEvaluationTask(self):
		"""
		"""
		pass

	@abstractmethod
	def generateSummary(self):
		"""
		"""
		pass 
	@abstractmethod
	def doHouseKeeping(self):
		"""
		This method will close existing database connections and 
		other resouces it has used. 
		"""
		pass
################# Summary Generation Code #################################################
	def __constructSingleDocGraphP2V(self):
		graph = nx.Graph() 
		sortedSentenceDict = sorted(self.sentenceDict.items(),\
		 key=operator.itemgetter(0), reverse=True) 

		for node_id,value in sortedSentenceDict:
			for in_node_id, value in sortedSentenceDict:
				doc_vec_1 = self.vecDict[node_id]
				doc_vec_2 = self.vecDict[in_node_id]
				sim = np.inner(doc_vec_1, doc_vec_2)
				if 	sim > self.intraThrSummary: 
					graph.add_edge(node_id, in_node_id, weight=sim)

		return graph

	def __dumpSummmaryToTable(self, doc_id, prSummary, idMap, methodID):
		position = 1
		for sumSentID, value  in prSummary.getSummary(self.dumpingFactor):
			if 	methodID == 1:
				sumSentID = idMap [sumSentID]
			if  position > len(self.sentenceDict) or  position > math.ceil(len(self.sentenceDict) * self.topNSummary):
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
		nx_G = self.__constructSingleDocGraphP2V()
		prSummary = PageRankBasedSummarizer(nx_G = nx_G)
		self.__dumpSummmaryToTable(doc_id, prSummary, "", methodID)

	def __sumarizeAndWriteTFIDFBasedSummary(self, doc_id, methodID):
		wbasedGenerator = WordBasedGraphGenerator(\
			sentDictionary=self.sentenceDict,\
		 	threshold=self.intraThrSummary)
		nx_G, idMap = wbasedGenerator.generateGraph()

		prSummary = PageRankBasedSummarizer(nx_G = nx_G)
		self.__dumpSummmaryToTable(doc_id, prSummary, idMap, methodID)

	def populateSummary(self, methodID, vecDict):
		"""
		"""
		self.vecDict = vecDict
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
				if methodID >1:
					self.__summarizeAndWriteLatentSpaceBasedSummary(id_, methodID)
				else:
					self.__sumarizeAndWriteTFIDFBasedSummary(id_ , methodID)

########################### Evaluation Data Generation code ##################

	def writeClassificationData(self,result, fileToWrite, vecDict):
		for row_id in range(0, len(result)):
	 		id_ = result[row_id][0]
	 		topic = result[row_id][1]

	 		vec = vecDict[id_] 
	 		vec_str = ','.join(str(x) for x in vec)
	 		fileToWrite.write("%s,%s,%s%s"%(id_,vec_str,topic, os.linesep))

	def generateData(self,summaryMethodID, latReprName, vecDict):
		trainFileToWrite = open("%s/%strain_%i.csv"%(self.trainTestFolder,\
			 latReprName, summaryMethodID), "w")
		testFileToWrite = open("%s/%stest_%i.csv"%(self.trainTestFolder,\
			 latReprName, summaryMethodID), "w")

		for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'YES'"],\
			 	["sentence.topic","<>","'unsup'"] ], [], []):
				self.writeClassificationData (result, trainFileToWrite, vecDict)

		for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'NO'"] ], [], []):
			 	self.writeClassificationData (result, testFileToWrite, vecDict)
		
########################### Evaluation Report Code ###########################


	def getXY(self, data):
		"""
		This function assumes that the data (pandas DF) has id in the 
		first column, label in the last column and features 
		in the middle. It returns features as X and label as Y.
		"""
		X = data[data.columns[1: data.shape[1]-1 ]]
		Y = data[data.shape[1]-1]
		return (X, Y)
	
	
	def runClassificationTask(self, summaryMethodID, latReprName):
		"""
		This function uses the generated train and test 
		files to build a logistic regression model using 
		scikit-learn. It will save the results 
		into files.
		"""
		
		train = pd.read_csv("%s/%strain_%i.csv"%(self.trainTestFolder,\
			 latReprName, summaryMethodID), header=None)
		test = pd.read_csv("%s/%stest_%i.csv"%(self.trainTestFolder,\
			 latReprName, summaryMethodID), header=None)
									
		train_X, train_Y = self.getXY(train)
		test_X, test_Y = self.getXY(test)

##################### Logistic Regression ###########################			
		logistic = linear_model.LogisticRegression()
		logit = logistic.fit(train_X, train_Y)
			
		result = pd.DataFrame()
		result['predicted_values'] = logit.predict(test_X)
		result['true_values'] = test_Y
		result.to_csv("%s/%sresult_%i.csv"%(self.trainTestFolder,\
			latReprName, summaryMethodID), index=False)
			
		labels = set(result['true_values'])
		class_labels = {}
		for i, label in enumerate(labels):
			class_labels[label] = label
			
		evaluaiton = ClassificationEvaluation(result['true_values'], result['predicted_values'], class_labels)
		
		evaluationResultFile = open("%s/%seval_%i.txt"%(self.trainTestFolder,\
				latReprName, summaryMethodID), "w")
		evaluationResultFile.write("%s%s%s" %("######Classification Report######\n", \
					evaluaiton._getClassificationReport(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Accuracy Score######\n", \
					evaluaiton._getAccuracyScore(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Confusion Matrix######\n", \
					evaluaiton._getConfusionMatrix(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Cohen's Kappa######\n", \
					evaluaiton._getCohenKappaScore(), "\n\n"))
					
		Logger.logr.info("Evaluation with Logistic regression Completed.")
###################### Dummy Classifiers (Most Frequent) #######################################
		dummyClf = DummyClassifier(strategy='most_frequent',random_state=0)
		dummyClf.fit(train_X, train_Y)
		result['predicted_values'] = dummyClf.predict(test_X)
		evaluaiton = ClassificationEvaluation(result['true_values'],\
		 result['predicted_values'], class_labels)

		evaluationResultFile.write("%s%s%s" %("######Classification Report (Most Frequent) ######\n", \
					evaluaiton._getClassificationReport(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Accuracy Score######\n", \
					evaluaiton._getAccuracyScore(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Confusion Matrix######\n", \
					evaluaiton._getConfusionMatrix(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Cohen's Kappa######\n", \
					evaluaiton._getCohenKappaScore(), "\n\n"))

###################### Dummy Classifier () ######################################################
		dummyClf = DummyClassifier(strategy='stratified',random_state=0)
		dummyClf.fit(train_X, train_Y)
		result['predicted_values'] = dummyClf.predict(test_X)
		evaluaiton = ClassificationEvaluation(result['true_values'],\
		 result['predicted_values'], class_labels)

		evaluationResultFile.write("%s%s%s" %("######Classification Report (Stratified) ######\n", \
					evaluaiton._getClassificationReport(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Accuracy Score######\n", \
					evaluaiton._getAccuracyScore(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Confusion Matrix######\n", \
					evaluaiton._getConfusionMatrix(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Cohen's Kappa######\n", \
					evaluaiton._getCohenKappaScore(), "\n\n"))

