#!/usr/bin/env python

"""
A module to be used like a config file
"""

#-------the path to the stanford annotated files-------
dataPath = "data/stanford"

#-------the path to the training file--------
trainingFile = "data/training.csv"
trainingEnts = "data/entities_training.csv"

#-------the path to the dev file--------
devFile = "data/dev.csv"
devEnts = "data/entities_dev.csv"

#-------the path to the test file------
testFile = "data/testing.csv"
testEnts = "data/entities_testing.csv"

#don't skip any tokens
includeAll = False

#max sentence length
maxLen = 50
