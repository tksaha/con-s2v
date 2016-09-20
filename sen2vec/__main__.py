#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import random
import importlib
from io import open
from utility.ArgumentParserUtility  import ArgumentParserUtility


from documentReader.NewsGroupReader import NewsGroupReader 
from documentReader.ReutersReader import ReutersReader


from six import text_type as unicode

# from six import iteritems
# from six.moves import range
# import psutil


from log_manager.log_config import Logger 


module_dict ={"reuter": "documentReader.ReutersReader", 
		"news": "documentReader.NewsGroupReader" }
class_dict = {"reuter": "ReutersReader", 
		"news": "NewsGroupReader"}

def main():
	"""
	Logging levels can be: critical, error, warning, info, debug and notset
	We can Log as info, debug, warning, error and critical
	It Dynamic Loads corresonding dataset specific classes. ld 
	denotes load document into database or not 
	"""
	argparser = ArgumentParserUtility('Sen2Vec')
	argparser.add_argument_to_parser("dataset", "Please enter dataset "\
		"to work on [reuter, news]", True)
	argparser.add_argument_to_parser("ld", "Load into Database [0, 1]", True)
	argparser.parse_argument()

	
	dataset = argparser.get_value_of_argument("dataset")
	ld = argparser.get_value_of_argument("ld")

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
	reader.runBaselines()

if __name__ == "__main__":
   sys.exit(main())
