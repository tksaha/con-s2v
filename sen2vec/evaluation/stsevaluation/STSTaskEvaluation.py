#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math
import numpy as np
import pandas as pd
import scipy.stats 
import sklearn.metrics as mt
from sklearn import linear_model
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from sklearn.dummy import DummyClassifier
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse
from keras.models import Sequential
from keras.layers.core import Dense, Activation




class STSTaskEvaluation:
	"""
	Semantic Textual Similarity Task Evaluation Base
	"""
	__metaclass__ = ABCMeta

	def __init__(self):
		self.dataFolder = os.environ['TRTESTFOLDER']
		self.postgresConnection = kwargs['postgres_connection']
		self.rootdir = os.environ['SEN2VEC_DIR']
		#self.original_val = list ()
		#self.computed_val = list ()
		self.nclass = 5 
		self.max_rating = 5

	def computeAndWriteResults(self, latReprName):
		sp = scipy.stats.spearmanr(original_val,computed_val)[0]
		pearson = scipy.stats.pearsonr(original_val,computed_val)[0]

		evaluationResultFile = open("%s/%sstseval.txt"%(self.dataFolder,\
				latReprName), "w")
		evaluationResultFile.write("spearman corr:%.4f%s"%(sp, os.linesep))
		evaluationResultFile.write("pearsonn corr:%.4f%s"%(pearson, os.linesep))

	def getVectors (item_dict, vDict):
		X1, X2, Score = [], [], []
		for k,val in item_dict.items():
			Score.append(val)
			X1.append(vDict[k[0]])
			X2.append(vDict[k[1]])

		return X1, X2, Score 

	def runValidation(self, vDict, latReprName):
		Logger.logr.info ("[%s] Running Validation for STS Task"%())
		train_pair_file = open(os.path.join(self.rootdir,\
				"Data/train_pair_%s.p"%(os.environ['DATASET'])), "rb")
		train_dict = pickle.load(train_pair_file)
		trainA, trainB, Score = getVectors(train_dict, vDict)
		

		validation_pair_file = open(os.path.join(self.rootdir,\
				"Data/validation_pair_%s.p"%(os.environ['DATASET'])), "rb")
		val_dict = pickle.load(validation_pair_file)

		Logger.logr.info("Running Validation for %i dictionary Iterms"%len(val_dict))
		devA, devB , devScore = getVectors(train_dict, vDict)

		trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
	    devF = np.c_[np.abs(devA - devB), devA * devB]

	    trainY = encode_labels(Score)
	    devY = encode_labels(devScore)

	    lrmodel = prepare_model(ninputs=trainF.shape[1])
	    pearsonr, bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, devScore)

	    evaluationResultFile = open("%s/%sstseval_%i.txt"%(self.trainTestFolder,\
				latReprName, summaryMethodID), "w")
	    evaluationResultFile.write("Pearsonr = %.4f"%pearsonr)

	def runSTSTest(self, vDict, latReprName):
		train_pair_file = open(os.path.join(self.rootdir,\
				"Data/train_pair_%s.p"%(os.environ['DATASET'])), "rb")
		train_dict = pickle.load(train_pair_file)
		trainA, trainB, Score = getVectors(train_dict, vDict)
		

		validation_pair_file = open(os.path.join(self.rootdir,\
				"Data/validation_pair_%s.p"%(os.environ['DATASET'])), "rb")
		val_dict = pickle.load(validation_pair_file)

		Logger.logr.info("Running Validation for %i dictionary Iterms"%len(val_dict))
		devA, devB , devScore = getVectors(train_dict, vDict)

		trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
	    devF = np.c_[np.abs(devA - devB), devA * devB]

	    trainY = encode_labels(Score)
	    devY = encode_labels(devScore)

	    lrmodel = prepare_model(ninputs=trainF.shape[1])
	    pearsonr, bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, devScore)

	    ####### The above  will regenerate the validation best model ########## 
	    
		test_pair_file = open(os.path.join(self.rootdir,\
				"Data/test_pair_%s.p"%(os.environ['DATASET'])), "rb")
		test_dict = pickle.load(test_pair_file)

		Logger.logr.info("Running Test for %i dictionary Iterms"%len(test_dict))

		testA, testB, testScore = self.getVectors(test_dict, vDict)
		testF = np.c_[np.abs(testA - testB), testA * testB]

		r = np.arange(1,self.max_rating+1)
		yhat = np.dot(bestlrmodel.predict_proba(testF, verbose=2), r)
	    pr = pearsonr(yhat, testScore)[0]
	    sr = spearmanr(yhat, testScore)[0]
	    se = mse(yhat, testScore)

	    evaluationResultFile = open("%s/%sstseval_%i.txt"%(self.trainTestFolder,\
				latReprName, summaryMethodID), "w")
	    evaluationResultFile.write("Pearsonr = %.4f, "\
	    	" Spearmanr = %0.4f, MSE = "%(pearsonr, pr, se))

	def runSTSTest_TFIDF(self, vDict, latReprName):


    def prepare_model(ninputs=9600, nclass=5):
	    """
	    Set up and compile the model architecture 
	    (Logistic regression)
	    """
	    lrmodel = Sequential()
	    lrmodel.add(Dense(ninputs, nclass))
	    lrmodel.add(Activation('softmax'))
	    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam')
	    return lrmodel


	def train_model(lrmodel, X, Y, devX, devY, devscores):
	    """
	    Train model, using pearsonr on dev for early stopping
	    """
	    done = False
	    best = -1.0
	    r = np.arange(1,self.max_rating+1)
	    
	    while not done:
	        # Every 100 epochs, check Pearson on development set
	        lrmodel.fit(X, Y, verbose=2, shuffle=False, validation_data=(devX, devY))
	        yhat = np.dot(lrmodel.predict_proba(devX, verbose=2), r)
	        score = pearsonr(yhat, devscores)[0]
	        if score > best:
	            print score
	            best = score
	            bestlrmodel = copy.deepcopy(lrmodel)
	        else:
	            done = True

	    yhat = np.dot(bestlrmodel.predict_proba(devX, verbose=2), r)
	    score = pearsonr(yhat, devscores)[0]
	    #print 'Dev Pearson: ' + str(score)
	    return (score, bestlrmodel)
    

	def encode_labels(labels, nclass=5):
	    """
	    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
	    """
	    Y = np.zeros((len(labels), nclass)).astype('float32')
	    for j, y in enumerate(labels):
	        for i in range(nclass):
	            if i+1 == np.floor(y) + 1:
	                Y[j,i] = y - np.floor(y)
	            if i+1 == np.floor(y):
	                Y[j,i] = np.floor(y) - y + 1
	    return Y