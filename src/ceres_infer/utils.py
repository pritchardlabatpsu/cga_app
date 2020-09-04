#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utilities
@author: boyangzhao
"""

import pandas as pd
import re

def int2ordinal(n):
    #partially based on https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
    if (type(n) is int) or n.isdigit():
        if type(n) is not int:
            n = int(n)
        return "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
    else:
        return n
    
def getFeatGene(x, firstOnly = False):
    #get gene
    if pd.isnull(x):
        return ''
    
    r = re.findall('([^,\()]*)\s(\(([^,]*)\)\s)*\[([^,]*)\]',x)
    
    if firstOnly:
        return r[0][0]
    else:
        return [n[0] for n in r]

def getFeatSource(x, firstOnly = False):
    #get the data source
    if(pd.isnull(x)):
        return ''
    r = re.findall('[^,\()]*\s(\([^,]*\)\s)*\[([^,]*)\]',x)
    
    if firstOnly:
        return [n[1] for n in r][0]
    else:
        return [n[1] for n in r]