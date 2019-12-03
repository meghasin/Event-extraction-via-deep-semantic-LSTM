#!/usr/bin/env python

"""
A script to build document and word representations for the model, saving
them to disk
"""

from pickle import dump
import sys
import os
from os.path import basename

import config as c
import vectorize as v
from annotation import readEvents, readDocs, createInstances

def setupDataSet(dataPath, eventsFile, converters, includeAll):
	"""
	Preps the data for learning
	"""
	#read the event annotations
	events = readEvents(eventsFile)

	#read the data
	rawData, labels = createInstances(readDocs(dataPath, events), events, includeAll)

	#vectorize it
	data = v.vectorize(rawData, converters)

	return data, labels, events

def main(outFile):
	"""
	Prepares the data according the config sets and save it to disk as a
	pickled map
	"""
	#if not os.access(basename(outFile), os.W_OK):
		#raise Exception("Cannot write to {}".format(outFile))

	converters = [v.Word2VecFeats(v.loadW2V("data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"), 2),
	v.Doc2VecFeats("data/vectors/doc2vec/ace/doc_embeddings.txt"),
	v.Sentence2VecFeats("data/vectors/doc2vec/ace/sent_embeddings.txt")]

	#vectorize the data
	print("Read training")
	trainingData, trainingLabels, trainingEvents = setupDataSet(c.dataPath, c.trainingFile, converters, c.includeAll)

	print("Read dev")
	devData, devLabels, devEvents = setupDataSet(c.dataPath, c.devFile, converters, c.includeAll)

	print("Read testing")
	testData, testLabels, testEvents = setupDataSet(c.dataPath, c.testFile, converters, c.includeAll)

	data = {"train_x":trainingData, "train_y":trainingLabels, "dev_x":devData,
	"dev_y":devLabels, "test_x":testData, "test_y":testLabels, 
	"train_events":trainingEvents, "dev_events":devEvents, "test_events":testEvents,
	"info": "\n".join(map(str,converters))}
	
	out = open(outFile,"w")
	dump(data, out)
	out.close()

if __name__ == "__main__":
	main(sys.argv[1])
