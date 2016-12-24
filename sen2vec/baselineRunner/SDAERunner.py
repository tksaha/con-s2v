#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import gensim 
import numpy as np
import pandas as pd 
import sklearn.metrics as mt
from collections import Counter 
from log_manager.log_config import Logger 
from utility.Utility import Utility



class SDAERunner(BaselineRunner):