#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from baselineRunner.BaselineRunner import BaselineRunner


class TFIDFBaselineRunner(BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.latReprName = "TFIDF"
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.postgresConnection.connectDatabase()
        self.methodID = 1

    def prepareData(self, pd):
        """
        """
        pass 

    def runTheBaseline(self, rbase):
        pass 


    def runEvaluationTask(self):
       
        vDict = {}
        summaryMethodID = 2
        
        if os.environ['EVAL']=='TEST' and os.environ['TEST_FOR']=='CLASS':
            self._runClassificationTF(summaryMethodID, "TFIDF", vDict)
        else:
            self._runClusteringTF(summaryMethodID, "TFIDF", vDict)




