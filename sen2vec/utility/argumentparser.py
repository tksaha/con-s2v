import argparse
from utility import * 

# This module facilitates argument parsing 
# and help text generation 
# capability from python code

class ArgumentParserUtility(Utility):
	"""Argument Parsing Utility"""
	def __init__(self,parser_description):
		self.parser = argparse.ArgumentParser(description=parser_description)
		self.arg_dictionary = {}
	def add_argument_to_parser(self, argument_name, help_text, required):
		self.parser.add_argument("-%s"%argument_name, "--%s"%argument_name, help=help_text, required=required, action='store')
	def parse_argument(self):
		self.arg_dictionary = vars(self.parser.parse_args());
	def get_value_of_argument(self, argument_name):
		return self.arg_dictionary['%s'%argument_name]