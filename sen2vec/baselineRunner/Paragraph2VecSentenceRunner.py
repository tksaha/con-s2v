#!/usr/bin/env python
# -*- coding: utf-8 -*-


from paragraph2vec.Para2Vec import Para2Vec
from log_manager.log_config import Logger 

class Paragraph2VecSentenceRunner(BaselineRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)
		self.sents = os.environ['P2VECSENTRUNNERINFILE']
		self.sentRepr = os.environ['P2VECSENTRUNNEROUTFILE']
	
	def prepareData(self):
		"""
		"""
		

	
	def runTheBaseline(self):
		"""
		"""
		pass

	
	def runEvaluationTask(self):
		"""
		"""
		pass

	
	def prepareStatisticsAndWrite(self):
		"""
		"""
		pass