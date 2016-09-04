#!/usr/bin/env python
# -*- coding: utf-8 -*-


from gensim.models import Doc2Vec
import gensim.models.doc2vec
from log_manager.log_config import Logger 
import multiprocessing
from baselineRunner.BaselineRunner import BaselineRunner


assert gensim.models.doc2vec.FAST_VERSION > -1, \
	"this will be painfully slow otherwise"

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(filename)):
            yield LabeledSentence(words=line.split(),\
             labels=['SENT_%s' % uid])


class Paragraph2VecSentenceRunner(BaselineRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)
		self.sents = os.environ['P2VECSENTRUNNERINFILE']
		self.sentRepr = os.environ['P2VECSENTRUNNEROUTFILE']
		self.cores = multiprocessing.cpu_count()

	
	def prepareData(self):
		"""
		Query Sentence Data. As a rough heuristics 
		sentences shorter than 9 words are excluded. We dump 
		both the sentence and their ids in different files. 
		"""
		self.postgresConnection.connect_database()
		
		sentfiletoWrite = open(self.sents,"w", encoding='utf-8', errors='ignore')
		
		for result in memoryEfficientSelect(["id","content"],\
			 ["sentence"], [], [], ["id"]):
			for row_id in range(0,len(result)):
				id_ = result[row_id][0]
				content= result[row_id][1]
				if len(content.split(" ")) < 5:
					continue 
				else:
					sentfiletoWrite.write("%s,%s%s"\
						%(gensim.utils.to_unicode(content.lower()),id_,os.linesep))		
		sentfiletoWrite.close()
	
	def runTheBaseline(self):
		"""
		"""
		para2vecModel = Doc2Vec(LabeledSentence(self.sents), size=100,\
			 window=8, min_count=4, workers=self.cpu_count)
		para2vecmodel.save('%s_model.doc2vec' %(self.sents))
	
	def runEvaluationTask(self):
		"""
		"""
		pass

	
	def prepareStatisticsAndWrite(self):
		"""
		"""
		pass