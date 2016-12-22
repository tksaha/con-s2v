#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os 
import sys
import pickle
import gensim 
import numpy as np
import pandas as pd 
from sklearn import linear_model
import sklearn.metrics as mt
from abc import ABCMeta, abstractmethod
from collections import Counter 
from keras.utils import np_utils
from keras import backend as K
from log_manager.log_config import Logger 
from keras.layers import Embedding
from keras.models import Sequential
from multiprocessing import Process
from keras.preprocessing import sequence
from utility.Utility import Utility
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution1D, GlobalMaxPooling1D
from db_connector.PostgresPythonConnector import PostgresPythonConnector


class SupervisedBaselineRunner:
    def __init__(self, dbstring, **kwargs):
        """
        """
        self.postgresConnection = PostgresPythonConnector(dbstring)


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

    def countFreq (self, sentence):
        content = gensim.utils.to_unicode(sentence) 
        content = self.utFunction.normalizeText(content, remove_stopwords=0)

        for token in content:
            self.word_counter[token] += 1

    def getlistOfIndexes(self, sentence):
        content = gensim.utils.to_unicode(sentence) 
        content = self.utFunction.normalizeText(content, remove_stopwords=0)

        list_of_indexes = []
        for token in content:
            if token in self.filtered_words:
                list_of_indexes.append(self.filtered_words[token])

        #Logger.logr.info(list_of_indexes)
        return list_of_indexes

    def encodeLabel (self, Y):
        if  self.isfirstTimeEncoding:
            self.encoder.fit(Y)
            self.isfirstTimeEncoding = False 
        
        encoded_Y = self.encoder.transform(Y)
        return encoded_Y

    def _getConfusionMatrix(self):
        return mt.confusion_matrix(self.true_values, self.predicted_values, labels = self.class_keys)
        
    def _getCohenKappaScore(self):
        return mt.cohen_kappa_score(self.true_values, self.predicted_values, labels = self.class_keys)
        
    def _getClassificationReport(self):
        return mt.classification_report(self.true_values, self.predicted_values,\
             labels = self.class_keys, target_names = self.class_names, digits=4)
    
    def _getAccuracyScore(self):
        return mt.accuracy_score(self.true_values, self.predicted_values)

    def _writeClassificationReport(self, evaluationResultFile, dummyName=""):

        evaluationResultFile.write("%s%s%s%s" %("######Classification Report",\
                    "(%s)######\n"%dummyName, \
                    self._getClassificationReport(), "\n\n"))
        evaluationResultFile.write("%s%s%s" %("######Accuracy Score######\n", \
                    self._getAccuracyScore(), "\n\n"))
        evaluationResultFile.write("%s%s%s" %("######Confusion Matrix######\n", \
                    self._getConfusionMatrix(), "\n\n"))
        evaluationResultFile.write("%s%s%s" %("######Cohen's Kappa######\n", \
                    self._getCohenKappaScore(), "\n\n"))
         evaluationResultFile.flush()
                    
        Logger.logr.info("Evaluation with Logistic regression Completed.")

    def getData (self, percent_vocab_size):
        # Leave id zero out of index, it is used for padding 
        summaryMethodID = 2
        train_sentences, train_labels = [], []
        test_sentences, test_labels  = [], []
        valid_sentences, valid_labels = [], []

    
        for result in self.postgresConnection.memoryEfficientSelect(["sentence.id", "content", "sentence.topic"],\
             ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
                ["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'YES'"]\
                 ], [], []):
                for row_id in range(0, len(result)):
                    train_sentences.append (result[row_id][1])
                    train_labels.append (result [row_id][2])
                    self.countFreq(result[row_id][1])
                    

        #Logger.logr.info (len(train_sentences))
        for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","content", "sentence.topic"],\
             ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
                ["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'NO'"] ], [], []):
                for row_id in range(0, len(result)):
                    test_sentences.append(result[row_id][1])
                    test_labels.append(result[row_id][2])
                    self.countFreq(result[row_id][1])

        #Logger.logr.info (len(train_sentences))  
        for result in self.postgresConnection.memoryEfficientSelect(["sentence.id", "content", "sentence.topic"],\
             ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
                ["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'VALID'"] ], [], []):
                for row_id in range(0, len(result)):
                    valid_sentences.append(result[row_id][1])
                    valid_labels.append(result[row_id][2])
                    self.countFreq(result[row_id][1])

        #Logger.logr.info (len(train_sentences)) 
        total_word = len(self.word_counter)
        total_to_take = int(total_word * percent_vocab_size) - 1 # minus 1 for keep first place for padding


        self.filtered_words = {}
        id_ = 1
        for word, count in self.word_counter.most_common(total_to_take):
            self.filtered_words[word] = id_
            id_ = id_ + 1

        # prepare data 
        self.tr_x = []
        for sentence in train_sentences:
            self.tr_x.append(self.getlistOfIndexes(sentence))

        self.ts_x = []
        for sentence in test_sentences:
            self.ts_x.append(self.getlistOfIndexes(sentence))

        self.val_x = []
        for sentence in valid_sentences: 
            self.val_x.append(self.getlistOfIndexes(sentence))

        self.tr_x = sequence.pad_sequences(self.tr_x, maxlen=self.maxlen)
        self.ts_x = sequence.pad_sequences(self.ts_x, maxlen=self.maxlen)
        self.val_x = sequence.pad_sequences(self.val_x, maxlen=self.maxlen)

        self.n_classes = len(np.unique(train_labels))

        self.tr_y  = np_utils.to_categorical(self.encodeLabel(train_labels), self.n_classes)
        self.val_y_prime = np_utils.to_categorical(self.encodeLabel(valid_labels), self.n_classes)
        self.ts_y  = self.encodeLabel(test_labels)
        self.val_y  = self.encodeLabel(valid_labels)

        self.max_features = total_to_take
        Logger.logr.info ("Total Number of Classes = %i" %self.n_classes)