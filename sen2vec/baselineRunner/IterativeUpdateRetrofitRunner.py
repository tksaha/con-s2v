#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import pickle 
import numpy as np 
import scipy.stats
import networkx as nx 
from log_manager.log_config import Logger 
from abc import ABCMeta, abstractmethod
from baselineRunner.BaselineRunner import BaselineRunner
from summaryGenerator.SummaryGenerator import SummaryGenerator
from retrofitters.IterativeUpdateRetrofitter import IterativeUpdateRetrofitter
from retrofitters.WeightedIterativeUpdateRetrofitter import WeightedIterativeUpdateRetrofitter
from retrofitters.RandomWalkIterativeUpdateRetrofitter import RandomWalkIterativeUpdateRetrofitter



class IterativeUpdateRetrofitRunner(BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.dataDir = os.environ['TRTESTFOLDER']
        self.latReprName = "iterative_update"
        self.postgresConnection.connectDatabase()
        self.sen2Vec = {}
        self.nIter = 20
        self.retrofittedsen2vReprFile = os.path.join(self.dataDir,"%s_repr"%self.latReprName)
        self.graphFile = os.path.join(self.dataDir, \
                "%s_graph_%s_%s"%(os.environ['DATASET'], os.environ['GINTERTHR'],\
                    os.environ['GINTRATHR']))
        self.p2vFile = os.path.join(self.dataDir, "%s_sentsCEXE_repr"%"p2vsent")
        self.Graph = nx.Graph()
        self.system_id = 5 
        # self.myalpha = float(os.environ['ITERUPDATE_ALPHA'])
    
    def prepareData(self, pd):
        """
        """
        pass 

    
    def runTheBaseline(self, rbase, latent_space_size):
        """
        Write down the Iterative update vector
        Hyperparameter numIter, alpha etc.
        """
        self.Graph = nx.read_gpickle(self.graphFile)
        p2vfileToRead = open ("%s.p" %self.p2vFile, "rb")
        self.sen2Vec = pickle.load(p2vfileToRead)


        Logger.logr.info("Dictionary has %i objects" % len(self.sen2Vec))
        if os.environ['EVAL']!= 'VALID':
            retrofitter = IterativeUpdateRetrofitter(numIter=self.nIter, nx_Graph = self.Graph) 
            retrofitted_dict, normalized_retrofitted_dict = retrofitter.retrofitWithIterUpdate(self.sen2Vec)
            iterupdatevecFile = open("%s_unweighted.p"%(self.retrofittedsen2vReprFile),"wb")
            iterupdatevecFile_Raw = open("%s_unweighted_raw.p"%(self.retrofittedsen2vReprFile),"wb")
            pickle.dump(retrofitted_dict, iterupdatevecFile)
            pickle.dump(normalized_retrofitted_dict, iterupdatevecFile_Raw)


        # wretrofitter = WeightedIterativeUpdateRetrofitter(numIter=20, nx_Graph = self.Graph)
        # retrofitted_dict, normalized_retrofitted_dict = wretrofitter.retrofitWithIterUpdate(self.sen2Vec, alpha = self.myalpha) #Hyperparameter
        # iterupdatevecFile = open("%s_weighted.p"%(self.retrofittedsen2vReprFile),"wb")
        # iterupdatevecFile_Raw = open("%s_weighted_raw.p"%(self.retrofittedsen2vReprFile),"wb")
        # pickle.dump(retrofitted_dict, iterupdatevecFile)
        # pickle.dump(normalized_retrofitted_dict, iterupdatevecFile_Raw)


        # randomwalkretrofitter = RandomWalkIterativeUpdateRetrofitter(numIter=10)
        # rand_retrofitted_dict, normalized_retrofitted_dict = randomwalkretrofitter.retrofitWithIterUpdate(self.sen2Vec)
        # rand_iterupdateFile = open("%s_randomwalk.p"%(self.retrofittedsen2vReprFile),"wb")
        # rand_iterupdateFile_Raw = open("%s_randomwalk_raw.p"%(self.retrofittedsen2vReprFile),"wb")
        # pickle.dump(rand_retrofitted_dict, rand_iterupdateFile)
        # pickle.dump(normalized_retrofitted_dict, rand_iterupdateFile_Raw)


    def generateSummary(self, gs, methodId, filePrefix, lambda_val=1.0, diversity=False):
        if gs <= 0: return 0
        itupdatevecFile = open("%s%s.p"%(self.retrofittedsen2vReprFile, filePrefix),"rb")
        itupdatevDict = pickle.load (itupdatevecFile)
        
        summGen = SummaryGenerator (diverse_summ=diversity,\
             postgres_connection = self.postgresConnection,\
             lambda_val = lambda_val)
        summGen.populateSummary(methodId, itupdatevDict)
    
    def __runEval(self, summaryMethodID, vecFileName, reprName):
        # vecFile = open("%s.p"%vecFileName,"rb")
        # vDict = pickle.load (vecFile)
        # self._runClassification(summaryMethodID, reprName, vDict)

        what_for =""
        try: 
            what_for = os.environ['VALID_FOR'].lower()
        except:
            what_for = os.environ['TEST_FOR'].lower()

        vDict  = {}
        if  "rank" in what_for:
            vecFile = open("%s.p"%(vecFileName),"rb")
            vDict = pickle.load(vecFile)
        else:
            vecFile_raw = open("%s_raw.p"%(vecFileName),"rb")
            vDict = pickle.load(vecFile_raw)

        Logger.logr.info ("Performing evaluation for %s"%what_for)
        self.performEvaluation(summaryMethodID, reprName, vDict)


    def runEvaluationTask(self):
        """
        Generate Summary sentences for each document. 
        Write sentence id and corresponding metadata 
        into a file. 
        """

        summaryMethodID = 2 
        if  os.environ['EVAL']!= 'VALID':
            vecFile = "%s_unweighted"%self.retrofittedsen2vReprFile
            reprName = "%s_unweighted"%self.latReprName
            self.__runEval(summaryMethodID, vecFile, reprName)
            
        # vecFile = "%s_weighted"%self.retrofittedsen2vReprFile
        # reprName = "%s_weighted"%self.latReprName
        # self.__runEval(summaryMethodID, vecFile, reprName)
        
        # vecFile = "%s_randomwalk"%self.retrofittedsen2vReprFile
        # reprName = "%s_randomwalk"%self.latReprName
        # self.__runEval(summaryMethodID, vecFile, reprName)

    def doHouseKeeping(self):
        """
        Here, we destroy the database connection.
        """
        self.postgresConnection.disconnectDatabase()
