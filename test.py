# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 14:03:11 2017

@author: Amir
"""

import csv

with open('data\questionnaire.csv','rb') as f:
    reader=csv.reader(f)
    for row in reader:
        print row[0]
        break
    
