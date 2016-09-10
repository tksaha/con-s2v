#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from log_manager.log_config import Logger 
import multiprocessing
from baselineRunner.BaselineRunner import BaselineRunner
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple


assert gensim.models.doc2vec.FAST_VERSION > -1, \
	"this will be painfully slow otherwise"
"""
https://docs.python.org/2/library/collections.html
"""
ReuterDocument = namedtuple('ReuterDocument', 'words tags')
label_sent = lambda id_: 'SENT_%s' %(id_)



def normalize_text(text):
	"""
	Replace breaks with spaces and then pad punctuation with spaces on both sides
	"""
	norm_text = text.lower().replace('<br />', ' ')
	for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
		norm_text = norm_text.replace(char, ' ' + char + ' ')
	return norm_text


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
				yield ReuterDocument(words=content.split(),\
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
				content = result[row_id][1].strip()
				if len(content.split()) < 9:
					n_nulls = 9 - len(content.split())
					for n in range(0,n_nulls):
						content = "NULL %s" %(content)
					#Logger.logr.info ("Size becomes %d"%len(content.split()))
				
				sent_dict = {}
				sent_dict["id"] = id_ 
				content = gensim.utils.to_unicode(content.lower())
				content = content.replace("\n", " ")
				content = normalize_text(content)
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
		for result in self.postgresConnection.memoryEfficientSelect(["id"],\
			 ["sentence"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]
				sen2vec_dict = {}
				sen2vec_dict["id"] = id_
				sen2vec_dict["vec"] = para2vecModel.docvecs[label_sent(id_)]
				pickle.dump(sen2vec_dict, sent2vecFile)
				
		sent2vecFile.close()
		self.postgresConnection.disconnect_database()
		return para2vecModel

	def runEvaluationTask(self):
		"""
		"""
		pass

	
	def prepareStatisticsAndWrite(self):
		"""
		"""
		pass