#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging.config
import logging.handlers

def singleton(cls):
	instances = {}
	def get_instance():
		if cls not in instances:
			instances[cls] = cls()
		return instances[cls]
	return get_instance()

@singleton
class Logger():
	"""
	For more details: 
	http://stackoverflow.com/questions/15727420/
	using-python-logging-in-multiple-modules
	"""
	def __init__(self): 
		LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"
		logging.basicConfig(format=LOGFORMAT)
		#logging.basicConfig(filename='logs.out',level=logging.INFO)
		self.logr = logging.getLogger('root')
		self.logr.setLevel(logging.INFO)

		

		