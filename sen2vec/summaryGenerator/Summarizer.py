#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod


class Summarizer:
	def __init__(self, *args, **kwargs):
		pass


	@abstractmethod
	def getSummary():
		pass