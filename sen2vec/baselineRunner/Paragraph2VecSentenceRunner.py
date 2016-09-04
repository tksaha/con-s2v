#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from log_manager.log_config import Logger 
import multiprocessing
from baselineRunner.BaselineRunner import BaselineRunner
from gensim.models.doc2vec import TaggedDocument


assert gensim.models.doc2vec.FAST_VERSION > -1, \
	"this will be painfully slow otherwise"

class TaggedLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename):
        	uid, _, content = line.strip().partition(",")
        	yield TaggedDocument(words=line.split(),tags=['SENT_%s' % uid])


class Paragraph2VecSentenceRunner(BaselineRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)
		self.sentsFile = os.environ['P2VECSENTRUNNERINFILE']
		self.sentReprFile = os.environ['P2VECSENTRUNNEROUTFILE']
		self.cores = multiprocessing.cpu_count()

	
	def prepareData(self):
		"""
		Query Sentence Data. As a rough heuristics 
		sentences shorter than 9 words are excluded. We dump 
		both the sentence and their ids in different files. 
		"""
		self.postgresConnection.connect_database()
		
		sentfiletoWrite = open(self.sentsFile,"w",\
			 encoding='utf-8', errors='ignore')
		
		for result in self.postgresConnection.memoryEfficientSelect(["id","content"],\
			 ["sentence"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]
				content= result[row_id][1]
				if len(content.split()) < 5:
					continue 
				else:
					content.replace(os.linesep, " ")
					sentfiletoWrite.write("%s,%s%s"\
						%(id_, gensim.utils.to_unicode(content.lower()),os.linesep))		
		sentfiletoWrite.close()
	
	def runTheBaseline(self):
		"""
		"""
		para2vecModel = Doc2Vec(TaggedLineSentence(self.sentsFile), size=100,\
			 window=8, min_count=4, workers=self.cores)
		para2vecModel.save('%s' %(self.sentReprFile))
	
	def runEvaluationTask(self):
		"""
		"""
		pass

	
	def prepareStatisticsAndWrite(self):
		"""
		"""
		pass