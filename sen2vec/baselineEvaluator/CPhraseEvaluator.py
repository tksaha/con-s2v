#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math 
import pickle
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from baselineEvaluator.BaselineEvaluator import BaselineEvaluator
from baselineRunner.CPhraseRunner import CPhraseRunner


class CPhraseEvaluator(BaselineEvaluator):
    def __init__(self, *args, **kwargs):
        """
        Skip Thought Preloaded Evaluator
        """
        BaselineEvaluator.__init__(self, *args, **kwargs)
        self.filePrefix = ""
        self.system_id_list = []

    def getOptimumParameters(self, f, optPDict, latent_space_size):
        self._setmetricString ()
        return optPDict

    def evaluateOptimum(self, pd, rbase, latent_space_size, optPDict, f):
        f.write("[CPhrase PreLoaded Baseline] (No Tuning) %s" %(os.linesep))   
        cphraseBaseline = CPhraseRunner(self.dbstring)
        self.system_id_list.append(cphraseBaseline.system_id)
        self.writeResults(pd, rbase, latent_space_size, cphraseBaseline, self.filePrefix, f)