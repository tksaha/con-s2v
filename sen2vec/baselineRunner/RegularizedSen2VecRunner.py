#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 
from gensim.models import Word2Vec
from log_manager.log_config import Logger 
from baselineRunner.BaselineRunner import BaselineRunner
import pickle
import math 
import operator 
import multiprocessing 
import subprocess 
import numpy as np 
from word2vec.WordDoc2Vec import WordDoc2Vec
from summaryGenerator.SummaryGenerator import SummaryGenerator
import scipy.stats

label_sent = lambda id_: 'SENT_%s' %(id_)


class RegularizedSen2VecRunner(BaselineRunner): 

    def __init__(self, *args, **kwargs):
        BaselineRunner.__init__(self, *args, **kwargs)
        self.regsen2vReprFile = os.environ["REGSEN2VECREPRFILE"]
        self.dataDir = os.environ['TRTESTFOLDER']
        self.sentsFile = os.environ['P2VCEXECSENTFILE']
        self.regBetaUNW = float(os.environ['REG_BETA_UNW'])
        self.regBetaW = float(os.environ['REG_BETA_W'])
        self.Graph = nx.Graph()
        self.window_size = str(10)
        self.cores = multiprocessing.cpu_count()
        self.graphFile = os.environ["GRAPHFILE"]
        self.latReprName = "reg_s2v"
        self.postgresConnection.connectDatabase()
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.system_id = 6
    
    def __getMaxNeighbors(self):
        """
        Calculates the maximum number of neighbors.
        """
        max_neighbor = 0 
        for nodes in self.Graph.nodes():
            nbrs = self.Graph.neighbors(nodes)
            if len(nbrs) > max_neighbor:
                max_neighbor = len(nbrs)

        return max_neighbor

    def __write_neighbors (self, max_neighbor, file_to_write, weighted):
        file_to_write.write("%s %s%s"%(self.Graph.number_of_nodes(),max_neighbor, os.linesep))

        for nodes in self.Graph.nodes():
            file_to_write.write("%s "%label_sent(str(nodes)))
            nbrs = self.Graph.neighbors(nodes)
            nbr_count = 0
            for nbr in nbrs:
                if weighted:
                    file_to_write.write("%s %s "%(label_sent(str(nbr)),self.Graph[nodes][nbr]['weight']))
                else:
                    file_to_write.write("%s %s "%(label_sent(str(nbr)),"1.0"))
                nbr_count = nbr_count +1 

            if nbr_count < max_neighbor:
                for  x in range(nbr_count, max_neighbor):
                    file_to_write.write("%s %s " %("-1","0.0"))

            file_to_write.write("%s"%os.linesep)

        file_to_write.flush()
        file_to_write.close()

    def prepareData(self, pd):
        """
        It prepares neighbor data for regularized sen2vec. 
        The first line of the file will indicate how nodes 
        are in the file and max number of neighbors. If a 
        particular node has less number of neighbors than the 
        maximum numbers then "-1" should be written as 
        neighbor. For the unweighted version, all weights should 
        be 1.0. 
        """
        if pd <= 0: return 0 
        self.Graph = nx.read_gpickle(self.graphFile)
        max_neighbor = self.__getMaxNeighbors()

        neighbor_file_w = open("%s_neighbor_w.txt"%(self.regsen2vReprFile), "w")
        neighbor_file_unw = open("%s_neighbor_unw.txt"%(self.regsen2vReprFile), "w")

        self.__write_neighbors (max_neighbor, neighbor_file_w, weighted=True)
        self.__write_neighbors (max_neighbor, neighbor_file_unw, weighted=False)
        self.Graph = nx.Graph()


    def __dumpVecs(self, reprFile, vecFile, vecRawFile):

        vModel = Word2Vec.load_word2vec_format(reprFile, binary=False)
        
        vec_dict = {}
        vec_dict_raw = {}

        for nodes in self.Graph.nodes():
            vec = vModel[label_sent(str(nodes))]
            vec_dict_raw[int(nodes)] = vec 
            vec_dict[int(nodes)] = vec /  ( np.linalg.norm(vec) +  1e-6)

        pickle.dump(vec_dict, vecFile)
        pickle.dump(vec_dict_raw, vecRawFile)

    def runTheBaseline(self, rbase, latent_space_size):
        if rbase <= 0: return 0 

        self.Graph = nx.read_gpickle(self.graphFile)

        wordDoc2Vec = WordDoc2Vec()
        wPDict = wordDoc2Vec.buildWordDoc2VecParamDict()
        wPDict["cbow"] = str(0) 
        wPDict["sentence-vectors"] = str(1)
        wPDict["min-count"] = str(0)
        wPDict["train"] = "%s.txt"%self.sentsFile
        wPDict["window"] = self.window_size
        
        wPDict["size"]= str(latent_space_size * 2)
        args = []

######################### Working for Weighted Neighbor File ################## 
        # neighborFile =  "%s_neighbor_w.txt"%(self.regsen2vReprFile)
        # wPDict["output"] = "%s_neighbor_w"%(self.regsen2vReprFile)
        # wPDict["neighborFile"], wPDict["beta"] = neighborFile, str(self.regBetaW)
        # args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 2)
        # self._runProcess (args)
        # self.__dumpVecs(wPDict["output"],\
        #      open("%s.p"%wPDict["output"], "wb"),\
        #      open("%s_raw.p"%wPDict["output"], "wb"))

        
######################### Working for UnWeighted Neighbor File ###################      
        neighborFile =  "%s_neighbor_unw.txt"%(self.regsen2vReprFile)
        wPDict["output"] = "%s_neighbor_unw"%(self.regsen2vReprFile)
        wPDict["neighborFile"], wPDict["beta"] = neighborFile, str(self.regBetaUNW)

        args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 2)
        self._runProcess (args)
        self.__dumpVecs(wPDict["output"],\
                open("%s.p"%wPDict["output"], "wb"),\
                open("%s_raw.p"%wPDict["output"], "wb"))
        self.Graph = nx.Graph()

    def generateSummary(self, gs, methodId, filePrefix,\
         lambda_val=1.0, diversity=False):
        if gs <= 0: return 0
        regsentvecFile = open("%s%s.p"%(self.regsen2vReprFile,\
             filePrefix),"rb")
        regsentvDict = pickle.load (regsentvecFile)
        
        summGen = SummaryGenerator (diverse_summ=diversity,\
             postgres_connection = self.postgresConnection,\
             lambda_val = lambda_val)

        summGen.populateSummary(methodId, regsentvDict)
        

    def __runEval(self, summaryMethodID, vecFileName, reprName):
        # vecFile = open("%s.p"%vecFileName,"rb")
        # vDict = pickle.load (vecFile)
        # self._runClassification(summaryMethodID, reprName, vDict)

        vecFile = open("%s_raw.p"%vecFileName, "rb")
        vDict = pickle.load (vecFile)
        Logger.logr.info("Regularized Dictionary has %i objects"%len(vDict))

        if os.environ['EVAL']=='VALID' and os.environ['VALID_FOR']=='CLASS':
            self._runClassificationValidation(summaryMethodID, "%s_raw"%reprName, vDict)
        elif os.environ['EVAL']=='VALID' and os.environ['VALID_FOR']=='CLUST':
            self._runClusteringValidation(summaryMethodID, "%s_raw"%reprName, vDict)
        elif os.environ['EVAL']=='TEST' and os.environ['TEST_FOR']=='CLASS':
            self._runClassification(summaryMethodID, "%s_raw"%reprName, vDict)
        else:
            self._runClustering(summaryMethodID, "%s_raw"%reprName, vDict)


    def evaluateRankCorrelation(self,dataset):
        vecFile =  open("%s%s.p"%(self.regsen2vReprFile,\
             "_neighbor_unw"),"rb")
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

            print (len(original_val))
            print (len(computed_val))
            sp = scipy.stats.spearmanr(original_val,computed_val)[0]
            pearson = scipy.stats.pearsonr(original_val,computed_val)[0]
            return sp, pearson

            
    def runEvaluationTask(self):
        summaryMethodID = 2 
        Logger.logr.info("Starting Regularized Sentence 2 Vector Evaluation")
        
        # regvecFile = "%s_neighbor_w"%(self.regsen2vReprFile)
        # reprName = "%s_neighbor_w"%self.latReprName
        # self.__runEval(summaryMethodID, regvecFile, reprName)

        regvecFile = "%s_neighbor_unw"%(self.regsen2vReprFile)
        reprName = "%s_neighbor_unw"%self.latReprName
        self.__runEval(summaryMethodID, regvecFile, reprName)

        
    def doHouseKeeping(self):
        """
        Here, we destroy the database connection.
        """
        self.postgresConnection.disconnectDatabase()