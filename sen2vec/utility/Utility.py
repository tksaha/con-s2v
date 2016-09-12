import os 
import sys 
from abc import ABCMeta, abstractmethod
import logging 
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer




class Utility:
	"""
	Utility Base
	"""
	def __init__ (self, utitlity_description):
		"""
		"""
		self.description = utitlity_description
		
	def normalize_text(self, text):
		"""
		Replace breaks with spaces and then pad punctuation with spaces on both sides
		"""
		norm_text = text.lower().replace('<br />', ' ')
		for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
			norm_text = norm_text.replace(char, ' ' + char + ' ')

		norm_text = norm_text.replace("\n", " ")
		stops = set(stopwords.words("english"))

		stemmer = SnowballStemmer("english")
		return [stemmer.stem(words) for words in norm_text.split() if words not in stops]