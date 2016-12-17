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
So, you should run p2v model and SeqItUpRetroRunner 
Simultaneously
"""

class SeqItUpRetroRunner(BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.dataDir = os.environ['TRTESTFOLDER']
        self.p2vFile = os.path.join(self.dataDir, "%s_sentsCEXE_repr"%"p2vsent")
        self.Graph = nx.Graph()
        self.postgresConnection.connectDatabase()
        self.sen2Vec = {}
        self.numIter = 20
        self.latReprName = "seq_iterative_update"
        self.system_id = 82
        self.seq_retr_vReprFile = os.path.join(self.dataDir, "%s_repr"%self.latReprName)


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

    
    # Latent space size is not used for this particular method
    def runTheBaseline(self, rbase, latent_space_size):
        """
        Write down the Iterative update vector
        Hyperparameter numIter, alpha etc.

        It assumes that, It has a vector file generated from sen2vec
        """
        p2vfileToRead = open ("%s.p" %self.p2vFile, "rb")
        self.sen2Vec = pickle.load(p2vfileToRead)

        Logger.logr.info("Dictionary has %i objects" % len(self.sen2Vec))
        if  os.environ['EVAL']!= 'VALID':
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
