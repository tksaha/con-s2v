#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import numpy as np
from log_manager.log_config import Logger 
import multiprocessing
from baselineRunner.BaselineRunner import Paragraph2VecSentenceRunner
from collections import namedtuple
from utility.Utility import Utility
import subprocess 
import pandas as pd
from sklearn import linear_model
from evaluation.classificationevaluaiton.ClassificationEvaluation import ClassificationEvaluation 



label_sent = lambda id_: 'SENT_%s' %(id_)


class P2VCExecutableRunner(Paragraph2VecSentenceRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		Paragraph2VecSentenceRunner.__init__(self, *args, **kwargs)
		self.sentsFile = os.environ['P2VCEXECSENTFILE']
		self.sentReprFile = os.environ['P2VCEXECOUTFILE']
		self.word2vecOut = os.environ['P2VECWORD2VECOUT']
		self.trainTestFolder = os.environ['P2VCEXECTRTESTFOLDER']
		self.word2vecExecutableDir= os.environ['WORD2VECEXECDIR']
	
	def prepareData(self):
		"""
		Query Sentence Data. We dump sentences with their sentence 
		ids. Pre-pad sentences with null word symbol if the number 
		of words in a sentence 
		is less than 9.
		"""
		sentfiletoWrite = open("%s.txt"%(self.sentsFile),"w")
		for result in self.postgresConnection.memoryEfficientSelect(["id","content"],\
			 ["sentence"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]
				content = gensim.utils.to_unicode(result[row_id][1].strip())
				content = self.utFunction.normalizeText(content, remove_stopwords=0)

				if len(content) < 9:
					n_nulls = 9 - len(content)
					for n in range(0,n_nulls):
						content.insert(0,"null")
				sentfiletoWrite.write("%s %s"%(label_sent(id_),' '.join(content)))
		sentfiletoWrite.close()
		

	def runTheBaseline(self, latent_space_size):
		"""
		We run the para2vec Model and then store sen2vec as pickled 
		dictionaries into the output file. 
		"""
		
		sent2vecFile = open("%s.p"%(self.sentReprFile),"wb")
		sen2vec_dict = {}

		args = [self.word2vecExecutableDir, "-train","%s.txt"%self.sentfiletoWrite,\
		    "-output",self.word2vecOut,\
			"-cbow",str(1),"-size", latent_space_size, "-window",str(8),\
			"-negative",str(5),"-hs",str(0),"-sample",str(1e-4) ,\
			"-threads",str(self.cores),\
			"-binary",str(0), "-iter",str(20),"-min_count",str(0),\
			"-sentence-vectors", str(1)]

		Logger.logr.info(args)
		proc = subprocess.Popen(args, out=stdout.PIPE, err=stderr.PIPE)
		out, err = proc.communicate()

		line_count = 0 
		for line in self.word2vecOut:
			if line_count == 0: 
				line_count = line_count + 1
				continue 
			line_elems = line.strip().split(" ")
			vec = np.zeros(latent_space_size, dtype=float)
			for pos in range(1, latent_space_size+1):
				vec[pos] = float(line_elems[pos]) 

			sen2vec_dict[int(line_elems[0])] = vec /  ( np.linalg.norm(vec) +  1e-6)


		Logger.logr.info("Total Number of Documents written=%i", len(sen2vec_dict))			
		pickle.dump(sen2vec_dict, sent2vecFile)
				
		sent2vecFile.close()