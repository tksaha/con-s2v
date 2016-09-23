#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import os 
import sys 
import networkx as nx 
import operator
import math 



import numpy as np 
from utility.Utility import Utility 
from collections import Counter 
from log_manager.log_config import Logger 


class WordBasedGraphGenerator:
	"""
	https://github.com/miso-belica/sumy/blob/dev/sumy/summarizers/lex_rank.py
	"""

	def __init__(self,*args, **kwargs):
		"""
		Pass a Dictionary sorted by the sentence id 
		"""
		self.sentenceDictionary = kwargs['sentDictionary']
		self.threshold = kwargs['threshold']

		self.sentenceDictionary = sorted(self.sentenceDictionary.items(), key=operator.itemgetter(0))
		self.id_mapping_dictionary ={}
		self.utFunction = Utility("Text normailization Utility")

		count = 0  
		for id_, value in self.sentenceDictionary:
			#Logger.logr.info("ID for dictionary %i" %id_)
			self.id_mapping_dictionary[count] = id_ 
			count = count + 1 



	def generateGraph(self):
		"""
		Sentence Dictionary will contain a sentence id 
		and corresponding sentences. The order does 
		not matter here. 
		"""
		sentencesWords = []
		for id_,sentence in self.sentenceDictionary:
			#Logger.logr.info("ID for array =%i" %id_) 
			sentencesWords.append(self.utFunction.normalizeText(sentence, remove_stopwords=0))

		tf_metrics = self._compute_tf(sentencesWords)
		idf_metrics = self._compute_idf(sentencesWords)

		matrix = self._create_matrix(sentencesWords, self.threshold, tf_metrics, idf_metrics)
		return nx.from_numpy_matrix(matrix), self.id_mapping_dictionary


	def _compute_tf(self, sentences):
		tf_values = map(Counter, sentences)

		tf_metrics = []
		for sentence in tf_values:
			metrics = {}
			max_tf = self._find_tf_max(sentence)

			for term, tf in sentence.items():
				metrics[term] = tf / max_tf

			tf_metrics.append(metrics)

		return tf_metrics

	def _find_tf_max(self, terms):
		return max(terms.values()) if terms else 1

	def _compute_idf(self, sentences):
		idf_metrics = {}
		sentences_count = len(sentences)

		for sentence in sentences:
			for term in sentence:
				if term not in idf_metrics:
					n_j = sum(1 for s in sentences if term in s)
					idf_metrics[term] = math.log(sentences_count / (1 + n_j))

		return idf_metrics

	def _create_matrix(self, sentences, threshold, tf_metrics, idf_metrics):
		"""
		Creates matrix of shape |sentences|×|sentences|. We use the cosine 
		similarity as it is. It is not normalized by the degree of a particular 
		node. 
		"""
	  
		sentences_count = len(sentences)
		matrix = np.zeros((sentences_count, sentences_count))
		degrees = np.zeros((sentences_count, ))

		for row, (sentence1, tf1) in enumerate(zip(sentences, tf_metrics)):
			for col, (sentence2, tf2) in enumerate(zip(sentences, tf_metrics)):
				matrix[row, col] = self._compute_cosine(sentence1, sentence2, tf1, tf2, idf_metrics)
				if matrix[row][col] <= threshold:
					matrix[row][col] = 0

		return matrix

	def _compute_cosine(self, sentence1, sentence2, tf1, tf2, idf_metrics):
		common_words = frozenset(sentence1) & frozenset(sentence2)

		numerator = 0.0
		for term in common_words:
			numerator += tf1[term]*tf2[term] * idf_metrics[term]**2

		denominator1 = sum((tf1[t]*idf_metrics[t])**2 for t in sentence1)
		denominator2 = sum((tf2[t]*idf_metrics[t])**2 for t in sentence2)

		if denominator1 > 0 and denominator2 > 0:
			return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
		else:
			return 0.0
