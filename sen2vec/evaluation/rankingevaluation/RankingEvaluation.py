#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger
from db_connector.PostgresPythonConnector import PostgresPythonConnector
from nltk.tokenize import sent_tokenize

import numpy as np
import pandas as pd
import sklearn.metrics as mt
import math
import subprocess

class RankingEvaluation:
	"""
	RankingEvaluation Base
	"""
	__metaclass__ = ABCMeta

	"""
	This program assumes that the environment has the following variables
	ROUGE = ~/rouge-1.5.5/ROUGE-1.5.5.pl #path to rouge executable
	ROUGE_EVAL_HOME=~/rouge-1.5.5/data #path to rouge data
	SUMMARYFOLDER=................./sen2vec/Data/Summary/ #path to summary folder
	MODELSUMMARYFOLDER=............/sen2vec/Data/Summary/model #path to model generated summary folder
	SYSTEMSUMMARYFOLDER=.........../sen2vec/Data/Summary/system #path to system generated summary folder
	"""
	def __init__(self, models, systems):
		self.models = models
		self.systems = systems
		self.rouge = os.environ["ROUGE"]
		self.summary_dir = os.environ["SUMMARYFOLDER"]
		self.models_dir = os.environ["MODELSUMMARYFOLDER"]
		self.systems_dir = os.environ["SYSTEMSUMMARYFOLDER"]
		
		self.dbstring = os.environ["DUC_DBSTRING"]
		self.postgresConnection = PostgresPythonConnector(self.dbstring)
		self.postgresConnection.connectDatabase()
		
		self.evalsDict = {}
	
	def __prepareConfigurationFile(self):
		config_file_name = ""
		for model in self.models:
			config_file_name += str(model)+"_"
		for system in self.systems:
			config_file_name += "_"+str(system)
		config_file_name += ".config"
		
		with open('%s%s%s' %(self.summary_dir,"/",config_file_name), 'w') as f:
			f.write('<ROUGE-EVAL version="1.5.5">%s' %os.linesep)

			for eval_ in self.evalsDict:
				f.write('<EVAL ID="%s">%s' %(eval_, os.linesep))
				
				f.write('<PEER-ROOT>%s</PEER-ROOT>%s' %(self.systems_dir, os.linesep))
				f.write('<MODEL-ROOT>%s</MODEL-ROOT>%s' %(self.models_dir, os.linesep))
				f.write('<INPUT-FORMAT TYPE="SEE"></INPUT-FORMAT>%s' %os.linesep)
				
				f.write('<PEERS>%s' %os.linesep)
				for system_ in self.evalsDict[eval_]['systems']:
					f.write('<P ID="%s">%s_%s.html</P>%s' %(system_, system_, eval_, os.linesep))
				f.write('</PEERS>%s' %os.linesep)
				
				f.write('<MODELS>%s' %os.linesep)
				for model_ in self.evalsDict[eval_]['models']:
					f.write('<M ID="%s">%s_%s.html</M>%s' %(model_, model_, eval_, os.linesep))
				f.write('</MODELS>%s' %os.linesep)
				
				f.write('</EVAL>%s' %os.linesep)
			f.write('</ROUGE-EVAL>')
		return config_file_name
	
	def __runRouge(self):
		config_file_name = self.__prepareConfigurationFile()
#		cmd = self.rouge + " -f A -a -x -s -m -2 -4 -u " + self.summary_dir+"/"+config_file_name
		cmd = self.rouge + " -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a " + self.summary_dir+"/"+config_file_name
		output = subprocess.check_output(cmd, shell=True)
		with open('%s%s%s' %(self.summary_dir, "/", config_file_name.replace('config', 'output')), 'w') as f:
			f.write(output)

	def __prepareSystemSummaryFiles(self):
		for system in self.systems:
			document_ids = []
			for result in self.postgresConnection.memoryEfficientSelect(\
							['distinct(doc_id)'],['summary'],[['method_id', '=', system]],[],[]):
				for row_id in range(0,len(result)):
					document_ids += [result[row_id][0]]
			
			for document_id in document_ids:
				sentences = []
				for result in self.postgresConnection.memoryEfficientSelect(\
				['content'], ['summary', 'sentence'], [['summary.sentence_id',\
				 '=', 'sentence.id'], ['summary.doc_id', '=',document_id], \
				 ['method_id', '=', system]], [], ['position']):
					for row_id in range(0,len(result)):
						sentences += [result[row_id][0]]
				
				for result in self.postgresConnection.memoryEfficientSelect(['filename'], ['document'], [['id', '=', document_id]], [], []):
					filename = result[0][0]

				if filename not in self.evalsDict:
					self.evalsDict[filename] = {'models': [], 'systems': []}
				self.evalsDict[filename]['systems'] += [system]
				
				systemfilename = "%s_%s.html" %(system, filename)
				with open('%s%s%s' %(self.systems_dir,"/",systemfilename), 'w') as f:
					f.write('<html>%s' %os.linesep)
					f.write('<head><title>%s</title></head>%s' %(systemfilename, os.linesep))
					f.write('<body bgcolor="white">%s' %os.linesep)
					for i, sentence in enumerate(sentences):
						i += 1
						sentence = sentence.replace(os.linesep, '')
						f.write('<a name="%s">[%s]</a> <a href="#%s" id=%s>%s</a>%s' %(i, i, i, i, sentence, os.linesep))
					f.write('</body>%s' %os.linesep)
					f.write('</html>%s' %os.linesep)

	def __prepareModelSummaryFiles(self):
		for model in self.models:
			filenames = []
			for result in self.postgresConnection.memoryEfficientSelect(\
							['filename'],['gold_summary'],[['method_id', '=', model]],[],[]):
				for row_id in range(0,len(result)):
					filenames += [result[row_id][0]]

			for filename in filenames:
				sentences = []
				for result in self.postgresConnection.memoryEfficientSelect(\
				['summary', 'metadata'], ['gold_summary'], [['gold_summary.filename', 'like', "'%s'" %filename], \
				['method_id', '=', model]], [], []):
					for row_id in range(0,len(result)):
						summarizer = result[row_id][1].split(':')[1]
						modelname = "%s.%s" %(model, summarizer)
						sentences = result[row_id][0]
						sentences = sent_tokenize(sentences)
						
						if filename not in self.evalsDict:
							self.evalsDict[filename] = {'models': [], 'systems': []}
						self.evalsDict[filename]['models'] += [modelname]
						
						modelfilename = "%s_%s.html" %(modelname, filename)
						with open('%s%s%s' %(self.models_dir,"/",modelfilename), 'w') as f:
							f.write('<html>%s' %os.linesep)
							f.write('<head><title>%s</title></head>%s' %(modelfilename, os.linesep))
							f.write('<body bgcolor="white">%s' %os.linesep)
							for i, sentence in enumerate(sentences):
								i += 1
								sentence = sentence.replace(os.linesep, '')
								f.write('<a name="%s">[%s]</a> <a href="#%s" id=%s>%s</a>%s' %(i, i, i, i, sentence, os.linesep))
							f.write('</body>%s' %os.linesep)
							f.write('</html>%s' %os.linesep)
	
	"""
	Protected Methods 
	"""
	def _getRankingEvaluation(self):
		self.__prepareSystemSummaryFiles()
		self.__prepareModelSummaryFiles()
		self.__runRouge()
		

