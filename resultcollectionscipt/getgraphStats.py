#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys 
import networkx as nx 

g= nx.read_gpickle(sys.argv[1])

print (g.number_of_nodes())
print (g.number_of_edges())