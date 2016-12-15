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
        self.retrofittedsen2vReprFile = os.environ["ITERUPDATESEN2VECFILE"]
        self.graphFile = os.environ["GRAPHFILE"]
        self.p2vFile = os.environ['P2VCEXECOUTFILE']
        self.myalpha = float(os.environ['ITERUPDATE_ALPHA'])
        self.Graph = nx.Graph()
        self.postgresConnection.connectDatabase()
        self.sen2Vec = {}
        self.nIter = 20
        self.latReprName = "iterativeupdate"
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.system_id = 5 
        
    
    def prepareData(self, pd):
        """
        """
        pass 

    
    def runTheBaseline(self, rbase):
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

        vecFile = open("%s_raw.p"%vecFileName, "rb")
        vDict = pickle.load (vecFile)
        if os.environ['EVAL']=='VALID' and os.environ['VALID_FOR']=='CLASS':
            self._runClassificationValidation(summaryMethodID, "%s_raw"%reprName, vDict)
        elif os.environ['EVAL']=='VALID' and os.environ['VALID_FOR']=='CLUST':
            self._runClusteringValidation(summaryMethodID, "%s_raw"%reprName, vDict)
        elif os.environ['EVAL']=='TEST' and os.environ['TEST_FOR']=='CLASS':
            self._runClassification(summaryMethodID, "%s_raw"%reprName, vDict)
        else:
            self._runClustering(summaryMethodID, "%s_raw"%reprName, vDict)


    def evaluateRankCorrelation(self,dataset):
        vecFile = open("%s%s.p"%(self.retrofittedsen2vReprFile, "_unweighted"),"rb")
        vDict = pickle.load (vecFile)


        if os.environ['EVAL']=='VALID':
            validation_pair_file = open(os.path.join(self.rootdir,"Data/validation_pair_%s.p"%(dataset)), "rb")
            val_dict = pickle.load(validation_pair_file)

            original_val = []
            computed_val = []
            for k, val in val_dict.items():
                original_val.append(val)
                computed_val.append(np.inner(vDict[(k[0])],vDict[(k[1])]))
            return scipy.stats.spearmanr(original_val,computed_val)[0]
        else:
            test_pair_file = open(os.path.join(self.rootdir,"Data/test_pair_%s.p"%(dataset)), "rb")
            test_dict = pickle.load(test_pair_file)

            original_val = []
            computed_val = []
            for k, val in test_dict.items():
                original_val.append(val)
                computed_val.append(np.inner(vDict[(k[0])],vDict[(k[1])]))

            if os.environ['TEST_AND_TRAIN'] =="YES":
                train_pair_file = open(os.path.join(self.rootdir,"Data/train_pair_%s.p"%(dataset)), "rb")
                train_dict = pickle.load(train_pair_file)
                for k, val in train_dict.items():
                    original_val.append(val)
                    computed_val.append(np.inner(vDict[(k[0])],vDict[(k[1])]))

            Logger.logr.info (len(original_val))
            Logger.logr.info (len(computed_val))
            sp = scipy.stats.spearmanr(original_val,computed_val)[0]
            pearson = scipy.stats.pearsonr(original_val,computed_val)[0]
            return sp, pearson

    def runEvaluationTask(self):
        """
        Generate Summary sentences for each document. 
        Write sentence id and corresponding metadata 
        into a file. 
        We should put isTrain=Maybe for the instances which 
        we do not want to incorporate in training and testing. 
        For example. validation set or unsup set
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
