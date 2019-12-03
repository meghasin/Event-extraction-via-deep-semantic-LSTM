#!/usr/bin/env python

"""
A script to build document and word representations for the model, saving
them to disk
"""

from pickle import dump, load
import sys
import os
from os.path import join
from collections import defaultdict

import numpy as n

import config as c
from vectorize import vectorize
import vectorize as v
from annotation import readEvents, readDocs, createInstances, readEntities
from ml.util import mkdir

from ner import NERIndex

def setupDataSet(dataPath, eventsFile, windowConv, contextConvs):
	"""
	Preps the data for learning
	"""
	#read the event annotations
	events = readEvents(eventsFile)

	#read the data
	rawData, labels = createInstances(readDocs(dataPath, events), events)

	left = n.array([windowConv.convert(i) for i in rawData])

	#vectorize it
	right = vectorize(rawData, contextConvs)

	return (left, right), labels, [i.event for i in rawData]

def writeWindow(dataPath, labelFile, converters, wordConv, outPrefix):
	"""
	Creates and saves event data
	"""
	(dataLeft, dataRight), labels, events = setupDataSet(dataPath, labelFile, wordConv, converters)

	#print out shape info
	print("left shape {}".format(dataLeft.shape))
	print("right shape {}".format(dataRight.shape))

	with open(outPrefix.format("left"), "w") as leftOut, open(outPrefix.format("right"), "w") as rightOut, open(outPrefix.format("labels"), "w") as labelsOut:
		
		n.save(leftOut, dataLeft)
		n.save(rightOut, dataRight)
		n.save(labelsOut, labels)

	return events

def main(outDir):
	"""
	Prepares the data according the config sets and save it to disk as a
	pickled map
	"""
	print("Building Converters")
	glovePath = "data/vectors/glove/glove.6B.50d.txt"
	w2vPath = "data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"
	d2vPath = "data/vectors/doc2vec/ace/doc_embeddings.txt"
	s2vPath = "data/vectors/doc2vec/ace/sent_embeddings.txt"

	#w2vModel = v.loadW2V(w2vPath)
	#gloveModel = v.loadGlove(glovePath)

	entTrain = "data/entities_training.csv"
	entDev = "data/entities_dev.csv"
	entTest = "data/entities_testing.csv"

	entities = readEntities(entTrain) + readEntities(entDev) + readEntities(entTest)
	entFeats = v.EntityFeats(entities)
	posFeats = v.SparsePOSFeats(load(open("data/pos_tags.p")))
	depFeats = v.SparseDependencyFeats(load(open("data/dep_tags.p")))
	docFeats = v.SparseDocTypeFeats("data/doc_types.txt")

	wordIndex = load(open("data/word_index.p"))
	entityIndex = load(open("data/entity_map.p"))

	#TODO remove
	#w2v = v.Word2VecFeats(defaultdict(lambda: [0.0]))
	#dataPath = "/home/walker/Data/ace/tmp/"

	#leftConverter = v.Word2VecFeats(v.loadW2V("data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"), 5)
	#leftConverter = v.Word2VecFeats(v.loadGlove("data/vectors/glove/glove.6B.50d.txt"), 20)
	#leftConverter = v.WindowFeats([v.Word2VecFeats(v.loadGlove("data/vectors/glove/glove.6B.50d.txt")), v.NERFeats("data/vectors/ner/nerIndex.p")], 20)
	#leftConverter = v.WindowFeats([v.Word2VecFeats(v.loadGlove(glovePath)), v.PositionFeats()], 20)
	#leftConverter = v.WindowFeats([v.Word2VecFeats(v.loadGlove(glovePath))], 20)
	#leftConverter = v.WindowFeats([v.Word2VecFeats(w2vModel)], 10)
	#leftConverter = v.WindowFeats([v.Word2VecFeats(gloveModel), entFeats, posFeats, depFeats], 20)
	#leftConverter = v.WindowFeats([v.Word2VecFeats(w2vModel), entFeats, v.PositionFeats()], 15)
	leftConverter = v.WindowFeats([v.WordEmbeddingFeats(wordIndex), v.EntityEmbeddingFeats(entityIndex, entities), 
	v.DistanceEmbeddingFeats()], 15)

	#rightConverters = [v.Word2VecFeats(v.loadW2V(w2vPath), 1),
	#v.Doc2VecFeats(d2vPath),
	#v.Sentence2VecFeats(s2vPath)]
	
	#rightConverters = [v.Word2VecFeats(w2vModel, 1), v.Doc2VecFeats(d2vPath), v.Sentence2VecFeats(s2vPath)]
	#rightConverters = [v.Word2VecFeats(gloveModel, 1), v.Word2VecFeats(w2vModel, 1), 
	#v.Doc2VecFeats(d2vPath), v.Sentence2VecFeats(s2vPath), entFeats, depFeats, posFeats, docFeats]

	rightConverters = [v.Doc2VecFeats(d2vPath), v.Sentence2VecFeats(s2vPath)]
	mkdir(outDir)

	#vectorize the data
	print("Read training")
	trainingEvents = writeWindow(c.dataPath, c.trainingFile, rightConverters, leftConverter, join(outDir, "training_{}.p"))

	print("Read dev")
	devEvents = writeWindow(c.dataPath, c.devFile, rightConverters, leftConverter, join(outDir, "dev_{}.p"))

	print("Read testing")
	testEvents = writeWindow(c.dataPath, c.testFile, rightConverters, leftConverter, join(outDir, "test_{}.p"))

	data = {"train_events":trainingEvents, "dev_events":devEvents, 
	"test_events":testEvents,
	"info": "\n".join(map(str,[leftConverter] + rightConverters))}
	
	with open(join(outDir, "info.p"),"w") as out:
		dump(data, out)

if __name__ == "__main__":
	main(sys.argv[1])
