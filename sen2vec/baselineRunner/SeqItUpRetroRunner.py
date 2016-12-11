#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import networkx as nx 
import pickle 
import numpy as np 
import scipy.stats
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger 
from baselineRunner.BaselineRunner import BaselineRunner
from summaryGenerator.SummaryGenerator import SummaryGenerator
from retrofitters.IterativeUpdateRetrofitter import IterativeUpdateRetrofitter
from retrofitters.WeightedIterativeUpdateRetrofitter import WeightedIterativeUpdateRetrofitter
from retrofitters.RandomWalkIterativeUpdateRetrofitter import RandomWalkIterativeUpdateRetrofitter



"""
This models assumes that you have a 
pretrained model. In our case, we use
s2v model as the pre-trained model.
"""

class SeqItUpRetroRunner(BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.p2vFile = os.environ['P2VCEXECOUTFILE']
        self.Graph = nx.Graph()
        self.postgresConnection.connectDatabase()
        self.sen2Vec = {}
        self.latReprName = "seq_iterative_update"
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.numIter = 20
        self.dataDir = os.environ['TRTESTFOLDER']
        self.seq_retr_vReprFile = os.path.join(self.dataDir, self.latReprName)

    def prepareData(self, pd):
        """
        Need to prepare the graph
        """

        for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
            ["document"], [], [], ["id"]):
            for row_id in range(0,len(doc_result)):
                document_id = doc_result[row_id][0]
                prev_sent_id = -1
                for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id'],['sentence'],[["doc_id","=",document_id]],[],['id']):
                    for inrow_id in range(0, len(sentence_result)):
                        sentence_id = int(sentence_result[inrow_id][0])
                        self.Graph.add_node(sentence_id)
                        if prev_sent_id < 0:
                            prev_sent_id = sentence_id
                        else:
                            self.Graph.add_edge(prev_sent_id, sentence_id)
                            prev_sent_id = sentence_id   


        Logger.logr.info("Total number of nodes in the Graph =%i"%self.Graph.number_of_nodes())     

    
    def runTheBaseline(self, rbase):
        """
        Write down the Iterative update vector
        Hyperparameter numIter, alpha etc.

        It assumes that, It has a vector file generated from sen2vec
        """
        
        p2vfileToRead = open ("%s.p" %self.p2vFile, "rb")
        self.sen2Vec = pickle.load(p2vfileToRead)


        Logger.logr.info("Dictionary has %i objects" % len(self.sen2Vec))
        if os.environ['EVAL']!= 'VALID':
            retrofitter = IterativeUpdateRetrofitter(numIter=self.numIter, nx_Graph = self.Graph) 
            retrofitted_dict, normalized_retrofitted_dict = retrofitter.retrofitWithIterUpdate(self.sen2Vec)
            iterupdatevecFile = open("%s_unweighted.p"%(self.seq_retr_vReprFile),"wb")
            iterupdatevecFile_Raw = open("%s_unweighted_raw.p"%(self.seq_retr_vReprFile),"wb")
            pickle.dump(retrofitted_dict, iterupdatevecFile)
            pickle.dump(normalized_retrofitted_dict, iterupdatevecFile_Raw)


    def generateSummary(self, gs, methodId, filePrefix, lambda_val=1.0, diversity=False):
        if gs <= 0: return 0
        itupdatevecFile = open("%s%s.p"%(self.seq_retr_vReprFile, filePrefix),"rb")
        itupdatevDict = pickle.load (itupdatevecFile)
        
        summGen = SummaryGenerator (diverse_summ=diversity,\
             postgres_connection = self.postgresConnection,\
             lambda_val = lambda_val)
        summGen.populateSummary(methodId, itupdatevDict)
    
    def __runEval(self, summaryMethodID, vecFileName, reprName):

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
        vecFile = open("%s%s.p"%(self.seq_retr_vReprFile, "_unweighted"),"rb")
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
        summaryMethodID = 2 
        if  os.environ['EVAL']!= 'VALID':
            vecFile = "%s_unweighted"%self.seq_retr_vReprFile
            reprName = "%s_unweighted"%self.latReprName
            self.__runEval(summaryMethodID, vecFile, reprName)
            

    def doHouseKeeping(self):
        """
        Here, we destroy the database connection.
        """
        self.postgresConnection.disconnectDatabase()
