#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import networkx as nx 

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import numpy as np 
from utility.Utility import Utility 




class WordBasedGraphGenerator:
	"""
	https://github.com/miso-belica/sumy/blob/dev/sumy/summarizers/lex_rank.py
	"""
	def __init__(self, *args, *kwargs):
		"""
		Pass a Dictionary sorted by the sentence id 
		"""
		self.sentenceDictionary = kwargs['sentDictionary']
		self.threshold = kwargs['threshold']


	def generateGraph:
		"""
		Sentence Dictionary will contain a sentence id 
		and corresponding sentences. The order does 
		not matter here. 
		"""
		sentencesWords = []
        sentences_words = [Utility.normalize_text(sentence) for id_,sentence in self.sentenceDictionary]
		
		tf_metrics = self._compute_tf(sentencesWords)
        idf_metrics = self._compute_idf(sentencesWords)

        matrix = self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)


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

    def _find_tf_max(terms):
        return max(terms.values()) if terms else 1

    def _compute_idf(sentences):
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
        Creates matrix of shape |sentences|×|sentences|.
        """
      
        sentences_count = len(sentences)
        matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count, ))

        for row, (sentence1, tf1) in enumerate(zip(sentences, tf_metrics)):
            for col, (sentence2, tf2) in enumerate(zip(sentences, tf_metrics)):
                matrix[row, col] = self._compute_cosine(sentence1, sentence2, tf1, tf2, idf_metrics)

                if matrix[row, col] > threshold:
                    matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    matrix[row, col] = 0

        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1

                matrix[row][col] = matrix[row][col] / degrees[row]

        return matrix

    def _compute_cosine(sentence1, sentence2, tf1, tf2, idf_metrics):
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