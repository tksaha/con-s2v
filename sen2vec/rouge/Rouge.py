#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import multiprocessing 


class Rouge:
	def __init__(self, *args, **kwargs):
		"""
		"""
		self.rougeParamDict = {}
		self.rougeEXE = os.environ["ROUGE"]
		self.cores = multiprocessing.cpu_count()

	def buildRougeParamDict(self):
		"""
		"""
		self.rougeParamDict['-c'] = str(95)    # confidence
		self.rougeParamDict['-2'] = str(-1)    # skip gram
		self.rougeParamDict['-r'] = str(1000)  # resampling
		self.rougeParamDict['-w'] = str(1.2)   # 
		self.rougeParamDict['-n'] = str(4)
		self.rougeParamDict['-l'] = str(20) # max length of words
		self.rougeParamDict['-m'] = str(1)  # stem 
		self.rougeParamDict['-s'] = str(1)  # remove stop words
		self.rougeParamDict['-a'] = str(1)
		self.rougeParamDict['conf'] =""
		return self.rougeParamDict

	def buildArgListforRouge(self, rpDict):	
		args = [self.rougeEXE, '-c', rpDict['-c'],\
			'-2', rpDict['-2'], '-r', rpDict['-r'],\
			'-U', '-n', rpDict['-n'], '-w', rpDict['-w'],\
			'-l',rpDict['-l']]

		if rpDict['-m'] == str(1):
			args.append('-m')

		if rpDict['-s'] == str(1):
			args.append('-s')
		args.append('-a')
		args.append(rpDict['conf'])

		return args