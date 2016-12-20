#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import pickle 
import gensim
import scipy.stats
from utility.Utility import Utility
from baselineRunner.BaselineRunner import BaselineRunner
from summaryGenerator.SummaryGenerator import SummaryGenerator


class TFIDFBaselineRunner(BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.system_id  = 1
        self.latReprName = "TFIDF"
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.postgresConnection.connectDatabase()
        self.utFunction = Utility("Text Utility")

    def prepareData(self, pd):
        pass 

    def runTheBaseline(self, rbase, latent_space_size):
        pass 

    def generateSummary(self, gs,  lambda_val=1.0, diversity=False):
        if gs <= 0: return 0
        summGen = SummaryGenerator (diverse_summ=diversity,\
             postgres_connection = self.postgresConnection,\
             lambda_val = lambda_val)

        summGen.populateSummary(1, {})

    def runEvaluationTask(self):
       
        vDict = {}
        summaryMethodID = 2
        
        self.performEvaluation(summaryMethodID, self.latReprName, vDict)


    def doHouseKeeping(self):
        self.postgresConnection.disconnectDatabase()