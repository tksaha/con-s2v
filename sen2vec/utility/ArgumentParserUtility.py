#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from utility.Utility import * 



class ArgumentParserUtility(Utility):
	"""
	Argument Parsing Utility: This module facilitates argument parsing 
	and help text generation capability from python code

	"""
	def __init__(self, *args, **kwargs):
		Utility.__init__(self, *args, **kwargs)
		self.parser = argparse.ArgumentParser(description=self.description)
		self.arg_dictionary = {}

	def add_argument_to_parser(self, argument_name, help_text, required):
		self.parser.add_argument("-%s"%argument_name, "--%s"%argument_name, 
			help=help_text, required=required, action='store')

	def parse_argument(self):
		self.arg_dictionary = vars(self.parser.parse_args())

	def get_value_of_argument(self, argument_name):
		return self.arg_dictionary['%s'%argument_name]