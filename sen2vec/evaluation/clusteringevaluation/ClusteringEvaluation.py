#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os 
import sys 
import math
import subprocess
import numpy as np
import pandas as pd
import sklearn.metrics as mt
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from db_connector.PostgresPythonConnector import PostgresPythonConnector
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans


class ClusteringEvaluation:
	"""
	Clustering Evaluation 
	"""
	def __init__(self, *args, **kwargs):
		self.dataFolder = os.environ['TRTESTFOLDER']
		self.postgresConnection = kwargs['postgres_connection']


	def __writeClusteringData(self, result, fileToWrite, vecDict):
		"""
		"""
		for row_id in range(0, len(result)):
	 		id_ = result[row_id][0]
	 		topic = result[row_id][1]

	 		vec = vecDict[id_] 
	 		vec_str = ','.join(str(x) for x in vec)
	 		fileToWrite.write("%s,%s,%s%s"%(id_,vec_str,topic, os.linesep))

	def generateData(self, summaryMethodID, latReprName, vecDict):
		datafileToWrite = open("%s/%sclusterData_%i.csv"%(self.dataFolder,\
			 latReprName, summaryMethodID), "w")
		
		for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", summaryMethodID]\
			 	 ], [], []):
				self.__writeClusteringData (result, datafileToWrite, vecDict)


	def generateDataValidation(self, summaryMethodID, latReprName, vecDict):
		datafileToWrite = open("%s/%sclusterData_%i.csv"%(self.dataFolder,\
			 latReprName, summaryMethodID), "w")
		
		for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", summaryMethodID],\
			 	["sentence.istrain","=","'VALID'"] ], [], []):
				self.__writeClusteringData (result, datafileToWrite, vecDict)


	def __getXY(self, data):
		"""
		This function assumes that the data (pandas DF) has id in the 
		first column, label in the last column and features 
		in the middle. It returns features as X and label as Y.
		"""
		X = data[data.columns[1: data.shape[1]-1 ]]
		Y = data[data.shape[1]-1]
		return (X, Y)


	def runClusteringTask(self, summaryMethodID, latReprName):

		Logger.logr.info("Doing Clustering Evaluation")
		data = pd.read_csv("%s/%sclusterData_%i.csv"%(self.dataFolder,\
			 latReprName, summaryMethodID), header=None)
		X, Y = self.__getXY(data)

		n_clusters=np.unique(Y)
		Logger.logr.info("Data Shape of the Clustering %s"%str(X.shape))
		Logger.logr.info("Number of clusters =%i" %len(n_clusters))

		estimator = KMeans(init='k-means++', n_clusters=len(np.unique(Y)), n_init=10)
		estimator.fit(X)

		evaluationResultFile = open("%s/%sclustereval_%i.txt"%(self.dataFolder,\
				latReprName, summaryMethodID), "w")
		evaluationResultFile.write("#######%s#############%s"%(latReprName,os.linesep))
		evaluationResultFile.write("HomoGeneity:%0.3f   Completeness:%.3f "\
			"  v_measure:%.3f   Adjusted Mutual Info Score:%.3f %s"\
    		% (mt.homogeneity_score(Y, estimator.labels_),\
    			mt.completeness_score(Y, estimator.labels_),\
    			mt.v_measure_score(Y, estimator.labels_),\
    			mt.adjusted_mutual_info_score(Y,  estimator.labels_), os.linesep))

	# Latent Representation name should pass "TFIDF"
	def runClusteringTaskTFIDF(self, summaryMethodID, latReprName): 
		from sklearn.feature_extraction.text import TfidfVectorizer
		
		test_corpus = []
		test_ids = []
		Y = []
		for result in self.postgresConnection.memoryEfficientSelect(["sentence.id",\
			 "sentence.content",	"sentence.topic"],\
			 ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
			 	["summary.method_id", "=", summaryMethodID]], [], []):
				for nrows in range(0,len(result)):
					test_ids.append(result[nrows][0])
					test_corpus.append(result[nrows][1])
					Y.append(result[nrows][2])
		
		
		vectorizer = TfidfVectorizer(stop_words='english')
		test_X = vectorizer.fit_transform(test_corpus)
		Logger.logr.info (test_X.shape)
		n_clusters=np.unique(Y)
		Logger.logr.info (n_clusters)

		estimator = KMeans(init='k-means++', n_clusters=len(np.unique(Y)), n_init=10)
		estimator.fit(test_X)				

		evaluationResultFile = open("%s/%sclustereval_%i.txt"%(self.dataFolder,\
				latReprName, summaryMethodID), "w")
		evaluationResultFile.write("#######%s#############%s"%(latReprName,os.linesep))
		evaluationResultFile.write("HomoGeneity:%0.3f   Completeness:%.3f "\
			"  v_measure:%.3f   Adjusted Mutual Info Score:%.3f %s"\
    		% (mt.homogeneity_score(Y, estimator.labels_),\
    			mt.completeness_score(Y, estimator.labels_),\
    			mt.v_measure_score(Y, estimator.labels_),\
    			mt.adjusted_mutual_info_score(Y,  estimator.labels_), os.linesep))