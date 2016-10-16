#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from math import log
from log_manager.log_config import Logger 

class InferSpace:
    """
    http://www.geeksforgeeks.org/dynamic-programming-set-32-word-break-problem/
    http://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
    """
    def __init__(self, *args, **kwargs):
        """
        Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability)
        """
        self.words = open("words-by-frequency.txt").read().split()
        self.wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
        self.maxword = max(len(x) for x in words)

    def infer_spaces(s):
        """
        Uses dynamic programming to infer the location of spaces in a string
        without spaces.
        """
        def best_match(i):
            """
            Find the best match for the i first characters, assuming cost has
            been built for the i-1 first characters.

            Returns a pair (match_cost, match_length).
            """
            candidates = enumerate(reversed(cost[max(0, i-self.maxword):i]))
            return min((c + self.wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)



        """
        Build the cost array.
        """
        cost = [0]
        for i in range(1,len(s)+1):
            c,k = best_match(i)
            cost.append(c)

        """
        Backtrack to recover the minimal-cost string.
        """
        out = []
        i = len(s)
        while i>0:
            c,k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])
            i -= k

        return " ".join(reversed(out))