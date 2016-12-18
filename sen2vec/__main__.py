#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import importlib
from io import open
from six import text_type as unicode
from log_manager.log_config import Logger 
from documentReader.RTReader import RTReader
from documentReader.SICKReader import SICKReader
from documentReader.IMDBReader import IMDBReader
from documentReader.ReutersReader import ReutersReader
from documentReader.NewsGroupReader import NewsGroupReader 
from utility.ArgumentParserUtility  import ArgumentParserUtility





module_dict ={"reuter": "documentReader.ReutersReader", "news": "documentReader.NewsGroupReader",
		"imdb": "documentReader.IMDBReader","rt": "documentReader.RTReader",
		"stree5way":"documentReader.SentimentTreeBankReader", "duc": "documentReader.DUCReader",
		"sick": "documentReader.SICKReader", "stree2way": "documentReader.SentimentTreeBank2WayReader"}
class_dict = {"reuter": "ReutersReader", "news": "NewsGroupReader","imdb": "IMDBReader",
		"rt": "RTReader", "stree5way": "SentimentTreeBank5WayReader", "duc": "DUCReader", "sick": "SICKReader", 
		 "stree2way": "SentimentTreeBank2WayReader"}

def main():
	"""
	Logging levels can be: critical, error, warning, info, debug and notset
	We can Log as info, debug, warning, error and critical
	It Dynamic Loads corresonding dataset specific classes. ld 
	denotes load document into database or not 
	"""
	argparser = ArgumentParserUtility('Sen2Vec')
	argparser.add_argument_to_parser("dataset", "Please enter dataset "\
		"to work on [reuter, news, imdb, stree2way, duc]", True)
	argparser.add_argument_to_parser("ld", "Load into Database 0 or 1", True)
	argparser.add_argument_to_parser("pd", "Prepare Data: 0 or 1", True)
	argparser.add_argument_to_parser("rbase", "Run the baseline 0 or 1", True)
	argparser.add_argument_to_parser("gs", "Generate Summary 0 or 1", True)
	argparser.parse_argument()

	
	dataset = argparser.get_value_of_argument("dataset")
	os.environ['DATASET'] = dataset
	if os.environ['DATASET'] == 'duc':
		os.environ['DATASET'] = "%s_%s"%(dataset, os.environ['DUC_TOPIC'])
	

	ld = argparser.get_value_of_argument("ld")
	pd = argparser.get_value_of_argument("pd")
	rbase = argparser.get_value_of_argument("rbase")
	gs = argparser.get_value_of_argument("gs")

	module = None

	try: 
		module = module_dict[dataset]
		klass = class_dict[dataset]
	except: 
		Logger.logr.error("Dataset Name does not match")
		sys.exit()

	Klass = getattr(importlib.import_module(module), klass)
	reader = Klass()
	Logger.logr.info("Successfuly loaded the class %s", str(Klass))
	
	reader.readDocument(int(ld)) 
	reader.runBaselines(int(pd), int(rbase), int(gs))

if __name__ == "__main__":
   sys.exit(main())