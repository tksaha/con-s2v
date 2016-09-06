#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
from log_manager.log_config import Logger 
import networkx as nx 

class IterativeUpdateRetrofitter:
	def __init__(self, *args, **kwargs):
		"""
		"""
		self.numIters = kwargs['numIters']
		self.sen2vecFilename = kwargs['sen2vecFilename']

	def readSentVecs(sen2vecFilename):
		"""
		Read the word vectors and normalize 
		"""
		Sen2Vectors = {}
		if sen2vecFilename.endswith('.gz'): 
			fileToRead = gzip.open(sen2vecFilename, 'r')
		else: 
			fileToRead= open(sen2vecFilename, 'r')
			for line in fileToRead:
				line = line.strip().lower()
				sentenceId = line.split()[0]
				Sent2Vectors[sentenceId] = numpy.zeros(len(line.split())-1, dtype=float)
				for index, vecVal in enumerate(line.split()[1:]):
      				sen2Vectors[sentenceId][index] = float(vecVal)
   
    		sen2Vectors[sentenceId] /= math.sqrt((sen2Vectors[sentenceId]**2).sum() + 1e-6)
    
  		Logger.logr.info("Vectors read from: %s %s"%(sen2vecFilename, os.linesep))
  		return sen2Vectors


	def retrofitWithIterUpdate(sen2vec, numIters):
		""" 
		Retrofit word vectors to a lexicon.
		alpha_i is set to number of neighbor of node i 
		Beta_ij is set to 1 

		"""

  		newSen2Vecs = deepcopy(sen2vec)
  		allSentenceIds = set(newSen2Vec.keys())
  		
  		for iter_ in range(numIters):
    		for sentenceId in allSentenceIds:
      			sentNeighbors = nx_Graph.neighbors[sentenceId]
      			numNeighbors = len(sentNeighbors)
      			if numNeighbors == 0:
        			continue

        		newVec = numNeighbors * sen2vec[sentenceId]
      			for neighborSentId in sentNeighbors:
        			newVec += newSen2Vecs[neighborSentId]

      			newSen2Vecs[sentenceId] = newVec/(2*numNeighbors)

  		return newSen2Vecs



  	def retrofit():
  		"""
  		"""
  		sen2vecDict  = readSentVecs(self.sen2vecFilename)
  		retroFitted = retrofitWithIterUpdate(sen2vecDict, self.numIters)
  		return retroFitted

  