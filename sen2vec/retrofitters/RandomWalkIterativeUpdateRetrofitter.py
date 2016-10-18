#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os 
import sys 
from log_manager.log_config import Logger 
import networkx as nx 
from copy import deepcopy
import numpy as np 

class RandomWalkIterativeUpdateRetrofitter:
    def __init__(self, *args, **kwargs):
      self.dataDir = os.environ['TRTESTFOLDER']
      self.walkFileName = "%s/node2vecwalk.txt"%(self.dataDir)
      self.numIters = kwargs['numIter']

    def retrofitWithIterUpdate(self, sen2vec):
      """
      Alpha_i is equal to 1
      Beta_ij is equal to 1/d_i
      """
      newSen2Vecs = deepcopy(sen2vec)
      normalized_newSen2Vecs = deepcopy(sen2vec)
      allSentenceIds = list(newSen2Vecs.keys())

      for iter_ in range(self.numIters):
        for line in open(self.walkFileName):
          walk = line.strip().split(" ")
          numNeighbors =  len(walk) - 1
        
          if numNeighbors == 0:
             continue 

          sentenceId = int(walk[0])
          
          newVec = numNeighbors * sen2vec[sentenceId]
          for neighborSentId in walk[1:]:
              neighbor = int(neighborSentId)
              newVec += newSen2Vecs[neighbor]
          newSen2Vecs[sentenceId] = newVec/(2*numNeighbors)

      for Id  in allSentenceIds:
        vec = newSen2Vecs[Id]    
        normalized_newSen2Vecs[Id] = vec / ( np.linalg.norm(vec) +  1e-6)

      return newSen2Vecs, normalized_newSen2Vecs
