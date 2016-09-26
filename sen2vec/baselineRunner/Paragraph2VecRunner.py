#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 

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


assert gensim.models.doc2vec.FAST_VERSION > -1, \
	"this will be painfully slow otherwise"
"""
https://docs.python.org/2/library/collections.html
"""
ImdbDocument = namedtuple('ImdbDocument', 'words tags')
label_doc = lambda id_: 'DOC_%s' %(id_)


class LineSentence(object):
	"""
	"""
	def __init__(self, filename):
		self.filename = filename 
	def __iter__(self):
		self.data_file=open(self.filename, 'rb')
		while True: 
			try:
				doc_dict = pickle.load(self.data_file)
				content = doc_dict ["content"]
				id_ = doc_dict["id"]
				yield ImdbDocument(words=content,\
					tags=[label_doc(id_)])
			except EOFError:
				break


class Paragraph2VecRunner(BaselineRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)
		self.docsFile = os.environ['P2VECRUNNERINFILE']
		self.docReprFile = os.environ['P2VECRUNNEROUTFILE']
		self.trainTestFolder = os.environ['TRTESTFOLDER']
		self.cores = multiprocessing.cpu_count()
		self.postgresConnection.connectDatabase()
		self.utFunction = Utility("Text Utility")
	
	def prepareData(self):
		"""
		Query Document Data. We dump both the Document and 
		their ids. Pre-pad sentences with 
		NULL word symbol if the number of words in a sentence 
		is less than 9.
		"""
		docfiletoWrite = open("%s.p"%(self.docsFile),"wb")
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
				
				doc_dict = {}
				doc_dict["id"] = id_ 
				doc_dict["content"] = content	
				pickle.dump(doc_dict,docfiletoWrite)
					
		docfiletoWrite.close()
		

	def runTheBaseline(self, latent_space_size):
		"""
		We run the para2vec Model and then store sen2vec as pickled 
		dictionaries into the output file. 
		"""
		para2vecModel = Doc2Vec(LineSentence("%s.p"%self.docsFile),\
			 size=latent_space_size,dm =0, iter=20, hs=0, negative=5,\
			 window=10, min_count=1, min_alpha=0.025, sample=1e-4, workers=self.cores)
		
		doc2vecFile = open("%s.p"%(self.docReprFile),"wb")
		doc2vec_dict = {}

		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			 ["document"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]	
				vec = para2vecModel.docvecs[label_doc(id_)]
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