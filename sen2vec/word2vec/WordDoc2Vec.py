#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 

class WordDoc2Vec: 
	def __init__(self, *args, **kwargs):
		self.wordParamDict = {}
		self.lineEXE = os.environ['LINEEXEFILE']
		self.doc2vecMIKOLOVExecutableDir= os.environ['DOC2VECEXECDIR']
		self.cores = multiprocessing.cpu_count()

	def buildWordDoc2VecParamDict(self):
		"""
		"""
		wordParamDict["train"] = ""
		wordParamDict["init"] = ""
		wordParamDict["output"] = ""
		wordParamDict["cbow"] = str(1)
		wordParamDict["size"] = str(300)
		wordParamDict["window"] = str(10)
		wordParamDict["negative"] = str(5)
		wordParamDict["hs"] = str(0)
		wordParamDict["sample"] = str(1e-4)
		wordParamDict["threads"] = str(cores * 2)
		wordParamDict["binary"] = str(0)
		wordParamDict["iter"] = str(20)
		wordParamDict["min-count"]= str(1)
		wordParamDict["sentence-vectors"] = str(0)
		if wordParamDict[cbow]== str(1):
			wordParamDict['alpha'] = str(0.05)
		else:
			wordParamDict['alpha'] = str(0.025)
		return self.wordParamDict

	def buildArgListforW2V(self, wPDict):
		args = [self.sent2vecMIKOLOVExecutableDir, "-train",wPDict["train"],\
		    "-output",wPDict["output"],\
			"-cbow",wPDict["cbow"],"-size",wPDict["size"], "-window",wPDict["window"],\
			"-negative",wPDict["negative"],"-hs",wPDict["hs"],"-sample",wPDict["sample"],\
			"-threads",wPDict["threads"],\
			"-binary",wPDict["binary"],"-iter",wPDict["iter"],"-min-count",wPDict["min-count"],\
			"-sentence-vectors", wPDict["sentence-vectors"]]
		return args 

	def buildArgListforW2VWithInit(self, wPDict):
		args = self.buildArgListforWord2Vec(self,wPDict)
		args.append["-init"]
		args.append[wPDict["init"]]
		return args



