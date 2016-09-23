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

class ClassificationEvaluation:
	"""
	ClassificationEvaluation Base
	"""
	__metaclass__ = ABCMeta

	"""
	true_values = A collection of true values
	predicted_values = A collection of predicted values
	class_labels = A dictionary of class ids/keys and names/values. {0: "People", 1: "Topic", 2: "Company", ...}
	"""
	def __init__(self, true_values, predicted_values, class_labels):
		self.true_values = true_values
		self.predicted_values = predicted_values
		self.class_keys = sorted(class_labels)
		self.class_names = [class_labels[key] for key in self.class_keys]
		pass

	"""
	Protected Methods 
	"""
	def _getConfusionMatrix(self):
		return mt.confusion_matrix(self.true_values, self.predicted_values, labels = self.class_keys)
		pass
		
	def _getCohenKappaScore(self):
		return mt.cohen_kappa_score(self.true_values, self.predicted_values, labels = self.class_keys)
		pass
		
	def _getClassificationReport(self):
		return mt.classification_report(self.true_values, self.predicted_values, labels = self.class_keys, target_names = self.class_names)
		pass
