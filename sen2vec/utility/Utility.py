import os 
import sys 
from abc import ABCMeta, abstractmethod
import logging 


class Utility:
	"""
	Utility Base
	"""
	def __init__ (self, utitlity_description):
		self.description = utitlity_description
		pass