#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import numpy as np
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from log_manager.log_config import Logger 
import multiprocessing
from baselineRunner.BaselineRunner import BaselineRunner
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
from utility.Utility import Utility 


assert gensim.models.doc2vec.FAST_VERSION > -1, \
	"this will be painfully slow otherwise"
"""
https://docs.python.org/2/library/collections.html
"""
ReuterDocument = namedtuple('ReuterDocument', 'words tags')
label_sent = lambda id_: 'SENT_%s' %(id_)


class LineSentence(object):
	"""
	"""
	def __init__(self, filename):
		self.filename = filename 
	def __iter__(self):
		self.data_file=open(self.filename, 'rb')
		while True: 
			try:
				sent_dict = pickle.load(self.data_file)
				content = sent_dict ["content"]
				id_ = sent_dict["id"]
				yield ReuterDocument(words=content,\
					tags=[label_sent(id_)])
			except EOFError:
				break


class Paragraph2VecSentenceRunner(BaselineRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)
		self.sentsFile = os.environ['P2VECSENTRUNNERINFILE']
		self.sentReprFile = os.environ['P2VECSENTRUNNEROUTFILE']
		self.cores = multiprocessing.cpu_count()
		self.postgresConnection.connect_database()
		self.utFunction = Utility("Text Utility")
	
	def prepareData(self):
		"""
		Query Sentence Data. As a rough heuristics 
		sentences shorter than 5 words are excluded. We dump 
		both the sentence and their ids in different files. 
		Prepad sentences with NULL word symbol if the number 
		of words in a particular sentence is less than 9.
		"""
		sentfiletoWrite = open("%s.p"%(self.sentsFile),"wb")
		for result in self.postgresConnection.memoryEfficientSelect(["id","content"],\
			 ["sentence"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]
				content = gensim.utils.to_unicode(result[row_id][1].strip())
				content = self.utFunction.normalize_text(content)

				if len(content) < 9:
					n_nulls = 9 - len(content)
					for n in range(0,n_nulls):
						content.insert(0,"NULL")
				
				sent_dict = {}
				sent_dict["id"] = id_ 
				sent_dict["content"] = content	
				pickle.dump(sent_dict,sentfiletoWrite)
					
		sentfiletoWrite.close()
		

	def runTheBaseline(self, latent_space_size):
		"""
		We run the para2vec Model and then store sen2vec as pickled 
		dictionaries into the output file. 
		"""
		para2vecModel = Doc2Vec(LineSentence("%s.p"%self.sentsFile),\
			 size=latent_space_size, window=8, min_count=1, workers=self.cores)
		
		sent2vecFile = open("%s.p"%(self.sentReprFile),"wb")
		sen2vec_dict = {}

		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			 ["sentence"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]	
				vec = para2vecModel.docvecs[label_sent(id_)]
				sen2vec_dict[id_] = vec /  ( np.linalg.norm(vec) +  1e-6)


		Logger.logr.info("Total Number of Documents written=%i", len(sen2vec_dict))			
		pickle.dump(sen2vec_dict, sent2vecFile)
				
		sent2vecFile.close()
		self.postgresConnection.disconnect_database()
		

	def runEvaluationTask(self):
		"""
		Generate Summary sentences for each document. 
		Write sentence id and corresponding metadata 
		into a file. 
		"""
		



	
	def prepareStatisticsAndWrite(self):
		"""
		"""
		pass