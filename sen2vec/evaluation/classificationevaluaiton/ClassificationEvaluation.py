#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from sklearn.dummy import DummyClassifier
from sklearn import linear_model
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import math

class ClassificationEvaluation:
	"""
	ClassificationEvaluation Base
	"""
	__metaclass__ = ABCMeta

	"""
	true_values = A collection of true values
	predicted_values = A collection of predicted values
	class_labels = A dictionary of class ids/keys and names/values. {0: "People", 1: "Topic", 2: "Company", ...}
	"""
	def __init__(self, *args, **kwargs):
		self.true_values = []
		self.predicted_values = []
		self.class_keys = {}
		self.class_names = []
		self.trainTestFolder = os.environ['TRTESTFOLDER']
		self.postgresConnection = kwargs['postgres_connection']

	"""
	Private Methods for evaluation 
	"""
	def _getConfusionMatrix(self):
		return mt.confusion_matrix(self.true_values, self.predicted_values, labels = self.class_keys)
		
	def _getCohenKappaScore(self):
		return mt.cohen_kappa_score(self.true_values, self.predicted_values, labels = self.class_keys)
		
	def _getClassificationReport(self):
		return mt.classification_report(self.true_values, self.predicted_values, labels = self.class_keys, target_names = self.class_names)
	
	def _getAccuracyScore(self):
		return mt.accuracy_score(self.true_values, self.predicted_values)

	def __writeClassificationData(self,result, fileToWrite, vecDict):
		"""
		"""
		for row_id in range(0, len(result)):
	 		id_ = result[row_id][0]
	 		topic = result[row_id][1]

	 		vec = vecDict[id_] 
	 		vec_str = ','.join(str(x) for x in vec)
	 		fileToWrite.write("%s,%s,%s%s"%(id_,vec_str,topic, os.linesep))

	def __writeClassificationReport(self, evaluationResultFile, dummyName=""):

		evaluationResultFile.write("%s%s%s%s" %("######Classification Report",\
					"(%s)######\n"%dummyName, \
					self._getClassificationReport(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Accuracy Score######\n", \
					self._getAccuracyScore(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Confusion Matrix######\n", \
					self._getConfusionMatrix(), "\n\n"))
		evaluationResultFile.write("%s%s%s" %("######Cohen's Kappa######\n", \
					self._getCohenKappaScore(), "\n\n"))
					
		Logger.logr.info("Evaluation with Logistic regression Completed.")


	def __getXY(self, data):
		"""
		This function assumes that the data (pandas DF) has id in the 
		first column, label in the last column and features 
		in the middle. It returns features as X and label as Y.
		"""
		X = data[data.columns[1: data.shape[1]-1 ]]
		Y = data[data.shape[1]-1]
		return (X, Y)
	
	"""
	Public methods 
	"""
	def generateData(self, summaryMethodID, latReprName, vecDict):
		trainFileToWrite = open("%s/%strain_%i.csv"%(self.trainTestFolder,\
			 latReprName, summaryMethodID), "w")
		testFileToWrite = open("%s/%stest_%i.csv"%(self.trainTestFolder,\
			 latReprName, summaryMethodID), "w")

		for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'YES'"],\
			 	["sentence.topic","<>","'unsup'"] ], [], []):
				self.__writeClassificationData (result, trainFileToWrite, vecDict)

		for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'NO'"] ], [], []):
			 	self.__writeClassificationData (result, testFileToWrite, vecDict)

	
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
									
		train_X, train_Y = self.__getXY(train)
		test_X, test_Y = self.__getXY(test)

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
			
		self.true_values =  result['true_values']
		self.predicted_values = result['predicted_values']
		self.class_keys = sorted(class_labels)
		self.class_names = [class_labels[key] for key in self.class_keys]
		evaluationResultFile = open("%s/%seval_%i.txt"%(self.trainTestFolder,\
				latReprName, summaryMethodID), "w")
		
		self.__writeClassificationReport(evaluationResultFile, latReprName)


###################### Dummy Classifiers (Most Frequent) #######################################
		# dummyClf = DummyClassifier(strategy='most_frequent',random_state=0)
		# dummyClf.fit(train_X, train_Y)
		# self.predicted_values  = dummyClf.predict(test_X)
		# self.__writeClassificationReport(evaluationResultFile, 'Most Frequent')

		
###################### Dummy Classifier () ######################################################
		dummyClf = DummyClassifier(strategy='stratified',random_state=0)
		dummyClf.fit(train_X, train_Y)
		self.predicted_values = dummyClf.predict(test_X)
		self.__writeClassificationReport(evaluationResultFile, 'Stratified')


