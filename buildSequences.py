#!/usr/bin/env python

from pickle import dump, load
import sys
from os.path import join

import config as c
import vectorize as v
from annotation import readEvents, readEntities, readDocs, createSequenceInstances, WordIndex
from ml.util import mkdir

from keras.preprocessing.sequence import pad_sequences
import numpy as n

def sparseWords(sequence, wordIndex, update):
	"""
	Makes a "sparse" vector, a vector of indexes
	"""
	results = []

	#for each word look up its index
	for inst in sequence:
		
		if update:
			index = wordIndex.updateIndex(inst.token.word)
		else:
			index = wordIndex.index(inst.token.word)

		results.append(index)

	return results

def vectorizeWordSequences(sequences, converters):
	"""
	Vectorzies the sequences
	"""
	results = []

	dim = -1

	#vectorize and pad each sequence
	for seq in sequences:

		matrix = v.vectorize(seq.instances, converters)
		hasShape = len(matrix.shape) > 1

		#get the shape of the 
		if hasShape:
			(length, dim) = matrix.shape	

		else:
			length = matrix.shape[0]

		pad = n.zeros( (c.maxLen - length, dim) )

		#if the matrix is totally empty then just use the padding
		if hasShape:
			matrix = n.vstack((matrix,pad))
		else:
			matrix = pad

		results.append(matrix)
	
	return n.array(results)

def vectorizeSequences(sequences, converters):
	"""
	Vectorzies the sequences
	"""
	results = []

	dim = -1

	#vectorize and pad each sequence
	for seq in sequences:

		#NOTE this assumes that the converters work on SequenceInstances
		results.append(n.concatenate([c.convert(seq) for c in converters]))
	
	return n.array(results, "float32")

def makeSequences(dataPath, eventsFile, entityFile, converters, wordConv, eventMap):
	"""
	Makes a data set comprised of sequences
	"""
	#NOTE: for BIO tagging of entities
	bio = True

	#read the event annotations
	events = readEvents(eventsFile)

	#read the entities
	entities = readEntities(entityFile)

	#read the data
	rawData, labels = createSequenceInstances(readDocs(dataPath, events), events, entities, eventMap, bio)

	words = vectorizeWordSequences(rawData, [wordConv])
	vec = vectorizeSequences(rawData, converters)

	return (words,vec), pad_sequences(labels, maxlen=c.maxLen, value=eventMap.nilIndex()), events, [s.toTag() for s in rawData]

def writeSequences(dataPath, labelFile, entityFile, converters, wordConv, eventMap, outPrefix):
	"""
	Creates and saves event data
	"""
	(dataLeft, dataRight), labels, events, tags = makeSequences(dataPath, labelFile, entityFile, converters, wordConv, eventMap)

	#print out shape info
	print("left shape {}".format(dataLeft.shape))
	print("right shape {}".format(dataRight.shape))

	with open(outPrefix.format("left"), "w") as leftOut, open(outPrefix.format("right"), "w") as rightOut, open(outPrefix.format("labels"), "w") as labelsOut:
		
		n.save(leftOut, dataLeft)
		n.save(rightOut, dataRight)
		n.save(labelsOut, labels)

	return events, tags

def main(outDir):
	"""
	Creates a tensor of sequences
	"""
	EVENT_MAP = "data/event_map.p"
	ENTITY_MAP = "data/entity_map.p"
	EVENT_ENTITY_MAP = "data/entity_event_map.p"
	ENTITY_MAP = "data/entity_map.p"
	d2vPath = "data/vectors/doc2vec/ace/doc_embeddings.txt"
	s2vPath = "data/vectors/doc2vec/ace/sent_embeddings.txt"
	w2vPath = "data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"
	glovePath = "data/vectors/glove/glove.6B.50d.txt"

	mkdir(outDir)

	w2vModel = v.loadW2V(w2vPath)
	#gloveModel = v.loadGlove(glovePath)

	wordConv = v.Word2VecFeats(w2vModel, 0)

	converters = [v.Doc2VecFeats(d2vPath), v.Sentence2VecFeats(s2vPath)]

	#make the event map
	#eventMap = load(open(EVENT_MAP))
	#eventMap = load(open(ENTITY_MAP))
	eventMap = load(open(EVENT_ENTITY_MAP))

	#vectorize the data
	print("Read training")
	trainingEvents, trainingTags = writeSequences(c.dataPath, c.trainingFile, c.trainingEnts, converters, wordConv, eventMap, join(outDir, "training_{}.p"))

	print("Read dev")
	devEvents, devTags = writeSequences(c.dataPath, c.devFile, c.devEnts, converters, wordConv, eventMap, join(outDir, "dev_{}.p"))

	print("Read testing")
	testEvents, testTags = writeSequences(c.dataPath, c.testFile, c.testEnts, converters, wordConv, eventMap, join(outDir, "test_{}.p"))
	
	data = {	"train_events":trainingEvents, "dev_events":devEvents, "test_events":testEvents,
	"info": "\n".join(map(str,converters)), "train_tags":trainingTags, "dev_tags":devTags, "test_tags":testTags}
	
	with open(join(outDir, "info.p"),"w") as infoOut:
		dump(data, infoOut)

if __name__ == "__main__":
	main(sys.argv[1])
