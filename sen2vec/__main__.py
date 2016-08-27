#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import random
from io import open
#from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter

#from concurrent.futures import ProcessPoolExecutor
import logging
from documentReader.newsgroupReader  import NewsGroupReader 


# from six import text_type as unicode
# from six import iteritems
# from six.moves import range


# import psutil
# from multiprocessing import cpu_count

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def main():
	newsReader = NewsGroupReader()
	print newsReader.readDocument()
	pass 

if __name__ == "__main__":
   sys.exit(main())
