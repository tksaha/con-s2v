#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger

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
	models = a list. [iterativeupdateunweighted, n2vecsent, p2vsent, n2vecsent_init, n2vecsent_retrofit, etc]
	This program assumes that the environment has the following variables
	ROUGE = ~/rouge-1.5.5/ROUGE-1.5.5.pl #path to rouge executable
	ROUGE_EVAL_HOME=~/rouge-1.5.5/data #path to rouge data
	SUMMARYFOLDER=................./sen2vec/Data/Summary/ #path to summary folder
	MODELSUMMARYFOLDER=............/sen2vec/Data/Summary/model #path to model generated summary folder
	SYSTEMSUMMARYFOLDER=.........../sen2vec/Data/Summary/system #path to system generated summary folder
	"""
	def __init__(self, models):
		self.models = models
		self.rouge = os.environ["ROUGE"]
		self.summary_dir = os.environ["SUMMARYFOLDER"]
		self.models_dir = os.environ["MODELSUMMARYFOLDER"]
		self.systems_dir = os.environ["SYSTEMSUMMARYFOLDER"]

	def __getSystemsEvals(self):
		systems_evals = []
		for root, dirs, files in os.walk(self.systems_dir):
			for file_ in files:
				if file_.endswith('.html'):
					systems_evals += [file_]

		evals = [system_eval.split('_')[1].split('.')[0] for system_eval in systems_evals] # It assumes system_eval.html convention
		systems = [system_eval.split('_')[0] for system_eval in systems_evals] # It assumes system_eval.html convention
		return (list(set(systems)), list(set(evals)))
	
	def __prepareConfigurationFile(self):
		config_file_name = '_'.join(self.models)+".config"
		with open('%s%s%s' %(self.summary_dir,"/",config_file_name), 'w') as f:
			f.write('<ROUGE-EVAL version="1.5.5">%s' %os.linesep)
			systems, evals = self.__getSystemsEvals()
			for eval_ in evals:
				f.write('<EVAL ID="%s">%s' %(eval_, os.linesep))
				
				f.write('<PEER-ROOT>%s</PEER-ROOT>%s' %(self.systems_dir, os.linesep))
				f.write('<MODEL-ROOT>%s</MODEL-ROOT>%s' %(self.models_dir, os.linesep))
				f.write('<INPUT-FORMAT TYPE="SEE"></INPUT-FORMAT>%s' %os.linesep)
				
				f.write('<PEERS>%s' %os.linesep)
				for system_ in systems:
					f.write('<P ID="%s">%s_%s.html</P>%s' %(system_, system_, eval_, os.linesep))
				f.write('</PEERS>%s' %os.linesep)
				
				f.write('<MODELS>%s' %os.linesep)
				for model_ in self.models:
					f.write('<M ID="%s">%s_%s.html</M>%s' %(model_, model_, eval_, os.linesep))
				f.write('</MODELS>%s' %os.linesep)
				
				f.write('</EVAL>%s' %os.linesep)
			f.write('</ROUGE-EVAL>')
		return config_file_name
	
	def __runRouge(self):
		config_file_name = self.__prepareConfigurationFile()
		cmd = self.rouge + " -f A -a -x -s -m -2 -4 -u " + self.summary_dir+"/"+config_file_name
		return subprocess.check_output(cmd, shell=True)


	"""
	Protected Methods 
	"""
	def _getRankingEvaluation(self):
		return self.__runRouge()

