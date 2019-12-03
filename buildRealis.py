#!/usr/bin/env python

"""
Builds a realis dataset from the Ace2005 event data
"""

from pickle import dump, load
from os.path import join
from collections import defaultdict
from argparse import ArgumentParser

import numpy as n

import config as c
from vectorize import vectorize
import vectorize as v
from annotation import readEvents, readDocs, createInstances, readEntities, ACTUAL, GENERIC, Event, Instance
from ml.util import mkdir


def instKey(instance):
	"""
	Returns the key for the instance
	"""
	return (instance.doc.id, instance.sentence.id, instance.token.id)

def mapInstances(instances):
	"""
	Maps instance keys into instances
	"""
	return {instKey(i):i for i in instances}

def instLemma(instance):
	return instance.token.lemma

def mapWordInstances(instances):
	"""
	Maps instances based on lemmas
	"""
	results = defaultdict(list)

	for inst in instances:
		results[instLemma(inst)].append(inst)

	return results

def matchInstances(docs, instances, includeAll=False):
	"""
	Returns a list of instances produced from the given set of documents
	"""
	results = []
	labels = []

	#map instances
	instMap = mapInstances(instances)

	wordMap = mapWordInstances(instances)

	#create instances for the tokens in each document
	for doc in docs:

		#for each sentence, make instances out of each token
		for sentence in doc.sentences:

			#for each token, if it is a noun, pronoun, verb, or adj make it an instance
			for token in sentence.tokens():

				if token.isVerb() or token.isNoun() or token.isPronoun() or token.isAdj() or includeAll:
				#if token.isVerb() or token.isNoun() or includeAll:
				#if True:

					#see if this token has an annotation
					key = (doc.id, sentence.id, token.id)
					instance = instMap[key]

					others = wordMap[instLemma(instance)]
					realis = None
					
					#if this instance is null and there are others
					if instance.isNil():
						
						#look for other non-nill instances
						events = [o for o in others if not o.isNil()]

						#create a negative instance 
						if len(events):
							realis = Event("REALIS-{}-{}-{}".format(doc.id, sentence.id, token.id), doc.id, GENERIC, sentence.id, token.id)

					else:

						#look for other non-nill instances
						nils = [o for o in others if o.isNil()]

						#create a positive instance 
						if len(nils):
							realis = Event("REALIS-{}-{}-{}".format(doc.id, sentence.id, token.id), doc.id, ACTUAL, sentence.id, token.id)

					if realis is not None:
						results.append(Instance(token, sentence, doc, realis))

						#add the label
						labels.append(realis.type)

	return results, labels

def setupDataSet(dataPath, eventsFile, windowConv):
	"""
	Preps the data for learning
	"""
	#read the event annotations
	events = readEvents(eventsFile)

	#read the data
	docs = readDocs(dataPath, events)

	#make instances
	rawData, _ = createInstances(docs, events)

	#look for potential realis instances
	realisInsts, labels = matchInstances(docs, rawData)

	#vectorize it
	left = n.array([windowConv.convert(i) for i in realisInsts])

	return left, labels, [i.event for i in realisInsts]

def writeWindow(dataPath, labelFile, wordConv, outPrefix):
	"""
	Creates and saves event data
	"""
	dataLeft, labels, events = setupDataSet(dataPath, labelFile, wordConv)

	#print out shape info
	print("left shape {}".format(dataLeft.shape))

	with open(outPrefix.format("left"), "w") as leftOut, open(outPrefix.format("labels"), "w") as labelsOut:
		
		n.save(leftOut, dataLeft)
		n.save(labelsOut, labels)

	return events

def main(outDir):
	"""
	Pairs up events with non-event to determine realis
	"""
	print("Building Converters")
	w2vPath = "data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"
	d2vPath = "data/vectors/doc2vec/ace/doc_embeddings.txt"
	s2vPath = "data/vectors/doc2vec/ace/sent_embeddings.txt"

	entTrain = "data/entities_training.csv"
	entDev = "data/entities_dev.csv"
	entTest = "data/entities_testing.csv"

	entities = readEntities(entTrain) + readEntities(entDev) + readEntities(entTest)
	entFeats = v.EntityFeats(entities)
	
	wordIndex = load(open("data/word_index.p"))
	entityIndex = load(open("data/entity_map.p"))

	leftConverter = v.WindowFeats([v.WordEmbeddingFeats(wordIndex), v.EntityEmbeddingFeats(entityIndex, entities), v.DistanceEmbeddingFeats()], 15)

	#rightConverters = [v.Word2VecFeats(v.loadW2V(w2vPath), 1),
	#v.Doc2VecFeats(d2vPath),
	#v.Sentence2VecFeats(s2vPath)]

	#rightConverters = [v.Word2VecFeats(w2vModel, 1), v.Doc2VecFeats(d2vPath), v.Sentence2VecFeats(s2vPath)]
	#rightConverters = [v.Word2VecFeats(gloveModel, 1), v.Word2VecFeats(w2vModel, 1), 
	#v.Doc2VecFeats(d2vPath), v.Sentence2VecFeats(s2vPath), entFeats, depFeats, posFeats, docFeats]

	#rightConverters = [v.Doc2VecFeats(d2vPath), v.Sentence2VecFeats(s2vPath)]
	rightConverters = []
	mkdir(outDir)

	#vectorize the data
	print("Read training")
	trainingEvents = writeWindow(c.dataPath, c.trainingFile, leftConverter, join(outDir, "training_{}.p"))

	print("Read dev")
	devEvents = writeWindow(c.dataPath, c.devFile, leftConverter, join(outDir, "dev_{}.p"))

	print("Read testing")
	testEvents = writeWindow(c.dataPath, c.testFile, leftConverter, join(outDir, "test_{}.p"))

	data = {"train_events":trainingEvents, "dev_events":devEvents,
	"test_events":testEvents,
	"info": "\n".join(map(str,[leftConverter] + rightConverters))}

	with open(join(outDir, "info.p"),"w") as out:
		dump(data, out)

if __name__ == "__main__":
	parser = ArgumentParser()

	parser.add_argument("path", help="The path to the output")

	args = parser.parse_args()

	main(args.path)
