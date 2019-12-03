#!/usr/bin/env python

"""
Makes a pickle file of the word to index
"""

import numpy as n
from pickle import dump

from ml.dependency import loadDocuments
from annotation import readEvents, TokenIndex
import config as c
import vectorize as v

current = 2
index = TokenIndex()

w2vFile = "data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"

#read the training events
events = readEvents(c.trainingFile)

#get the training docs
trainingDocs = set([e.docId for e in events])

#load the w2v weights
w2v = v.loadW2V(w2vFile)

#load the docs
for doc in loadDocuments(c.dataPath):

	#for all the training docs collect all the vocab 
	if doc.id in trainingDocs:
		
		print("Doc id {}".format(doc.id))

		for token in doc.tokens():

			index.updateIndex(token, w2v)

dump(index, open("data/word_index.p", "w"))
