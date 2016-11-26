#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import multiprocessing 

class WordDoc2Vec: 
	def __init__(self, *args, **kwargs):
		"""
		"""
		self.wordParamDict = {}
		self.doc2vecMIKOLOVExecutableDir= os.environ['DOC2VECEXECDIR']
		self.retrofitOneExe = os.environ['RETROFITONEEXE']
		self.regularizeSen2Vec = os.environ['REGSEN2VECEXE']
		#self.jointLearner = os.environ['JOINTLEXE']
		self.jointSupLearner = os.environ['JOINTSUPEXE']
		self.cores = multiprocessing.cpu_count()

	def buildWordDoc2VecParamDict(self):
		"""
		"""
		self.wordParamDict["train"] = ""
		self.wordParamDict["init"] = ""
		self.wordParamDict["output"] = ""
		self.wordParamDict["cbow"] = str(1)
		self.wordParamDict["size"] = str(300)
		self.wordParamDict["window"] = str(10)
		self.wordParamDict["negative"] = str(5)
		self.wordParamDict["hs"] = str(0)
		self.wordParamDict["sample"] = str(1e-4)
		self.wordParamDict["threads"] = str(self.cores * 10)
		self.wordParamDict["binary"] = str(0)
		self.wordParamDict["iter"] = str(20)
		self.wordParamDict["min-count"]= str(0)
		self.wordParamDict["sentence-vectors"] = str(0)
		if self.wordParamDict["cbow"]== str(1):
			self.wordParamDict["alpha"] = str(0.05)
		else:
			self.wordParamDict["alpha"] = str(0.025)
		return self.wordParamDict

	def buildArgListforW2V(self, wPDict, retrofit=0):
		if retrofit == 0:
			exeFile = self.doc2vecMIKOLOVExecutableDir
		elif retrofit == 1:
			exeFile = self.retrofitOneExe
		elif retrofit == 2:
			exeFile = self.regularizeSen2Vec
		elif retrofit == 3:
			exeFile = self.jointSupLearner

		args = [exeFile, "-train",wPDict["train"],\
		    "-output",wPDict["output"],\
			"-cbow",wPDict["cbow"],"-size",wPDict["size"], "-window",wPDict["window"],\
			"-negative",wPDict["negative"],"-hs",wPDict["hs"],"-sample",wPDict["sample"],\
			"-threads",wPDict["threads"],\
			"-binary",wPDict["binary"],"-iter",wPDict["iter"],"-min-count",wPDict["min-count"],\
			"-sentence-vectors", wPDict["sentence-vectors"]]
		return args 

	def buildArgListforW2VWithInit(self, wPDict, retrofit):
		args = self.buildArgListforW2V(wPDict, retrofit)
		args.append("-init")
		args.append(wPDict["init"])
		if retrofit == 1:
			args.append("-beta")
			args.append(wPDict["beta"])
		return args

	def buildArgListforW2VWithNeighbors(self, wPDict, retrofit):
		args = self.buildArgListforW2V(wPDict, retrofit)
		args.append("-neighbor")
		args.append(wPDict["neighborFile"])
		
		args.append("-lambda")
		args.append(wPDict['lambda'])
		
		args.append("-full-nbr")
		args.append(wPDict["full_data"])
		return args 

	def buildArgListforW2VWith_LAB_Neighbors(self, wPDict, retrofit):
		args = self.buildArgListforW2V(wPDict, retrofit)
		args.append("-neighbor")
		args.append(wPDict["neighborFile"])
		
		args.append("-label")
		args.append(wPDict["label"])

		args.append("-lambda")
		args.append(wPDict['lambda'])
		
		args.append("-full-nbr")
		args.append(wPDict["full_data"])
		return args; 
	




