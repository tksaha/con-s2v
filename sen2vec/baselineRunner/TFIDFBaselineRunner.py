#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import pickle 
import gensim
import scipy.stats
from utility.Utility import Utility
from baselineRunner.BaselineRunner import BaselineRunner



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

    def runTheBaseline(self, rbase):
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
        
        if os.environ['EVAL']=='TEST' and os.environ['TEST_FOR']=='CLASS':
            self._runClassificationTF(summaryMethodID, "TFIDF", vDict)
        else:
            self._runClusteringTF(summaryMethodID, "TFIDF", vDict)


    def doHouseKeeping(self):
        self.postgresConnection.disconnectDatabase()