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

    def retrofitWithIterUpdate(self, sen2vec):
      """
      Alpha_i is equal to number of neighbors 
      Beta_ij is equal to one 
      """
      newSen2Vecs = deepcopy(sen2vec)
      allSentenceIds = list(newSen2Vecs.keys())

      for line in open(self.walkFileName):
        walk = line.strip().split(" ")
        numNeighbors =  len(walk) - 1
        Logger.logr.info("Number of neighbors is %i"%numNeighbors)

        sentenceId = int(walk[0])
        newVec = numNeighbors * sen2vec[sentenceId]
        for neighborSentId in walk[1:]:
            newVec += newSen2Vecs[int(neighborSentId)]

        newSen2Vecs[sentenceId] = newVec/(2*numNeighbors)


      for Id  in allSentenceIds:
        vec = newSen2Vecs[Id] 
        newSen2Vecs[Id] = vec / ( np.linalg.norm(vec) +  1e-6)
        Logger.logr.info("Norm of the vector = %f"%np.linalg.norm(newSen2Vecs[Id]))

     
      return newSen2Vecs
