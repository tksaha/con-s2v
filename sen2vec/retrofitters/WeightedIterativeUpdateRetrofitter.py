#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os 
import sys 
from log_manager.log_config import Logger 
import networkx as nx 
from copy import deepcopy
import numpy as np 

class WeightedIterativeUpdateRetrofitter:
    def __init__(self, *args, **kwargs):
      self.numIters = kwargs['numIter']
      self.nx_Graph = kwargs['nx_Graph']

    def retrofitWithIterUpdate(self, sen2vec, alpha):
      """
      Please also check whether it is normalized?
      """
      newSen2Vecs = deepcopy(sen2vec)
      allSentenceIds = list(newSen2Vecs.keys())

      for iter_ in range(self.numIters):
        for sentenceId in allSentenceIds:
          sentNeighbors = self.nx_Graph.neighbors(sentenceId)
          numNeighbors = len(sentNeighbors)
          if numNeighbors == 0:
            continue
          newVec = alpha * sen2vec[sentenceId]
          total_weight = 0.0
          for neighborSentId in sentNeighbors:
            newVec += nx_Graph[sentenceId][neighborSentId]['weight'] * newSen2Vecs[neighborSentId]
            total_weight = total_weight + nx_Graph[sentenceId][neighborSentId]['weight']

          newSen2Vecs[sentenceId] = newVec/(alpha + total_weight)

      for Id  in allSentenceIds:
        vec = newSen2Vecs[Id] 
        newSen2Vecs[Id] = vec / ( np.linalg.norm(vec) +  1e-6)

      Logger.logr.info("Norm of the vector = %f"%np.linalg.norm(newSen2Vecs[allSentenceIds[0]]))
      return newSen2Vecs
