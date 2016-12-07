#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
import re
import numpy as np 
import gensim 
from baselineRunner.BaselineRunner import BaselineRunner
from log_manager.log_config import Logger 
from utility.Utility import Utility

class FastSentFHVersionRunner(BaselineRunner):
	
	def __init__(self, *args, **kwargs):
        BaselineRunner.__init__(self, *args, **kwargs)


 	def prepareData(self, pd):
 		pass 

 	def runTheBaseline(self, rbase, latent_space_size):
 		pass

 	def generateSummary():
 		pass

 	def runEvaluationTask():
 		pass

 	def doHouseKeeping():
 		pass 
