#!/usr/bin/env python

from annotation import readEvents, readDocs

dataPath = "/home/walker/Data/ace/full_annotated2/"
eventsFile = "data/training.csv"

#read the event annotations
events = readEvents(eventsFile)

docs = readDocs(dataPath, events)

vocab = set()

for doc in docs:
	for token in doc.tokens():
		vocab.add(token.lemma)

for lemma in vocab:
	try:
		print(lemma)
	except:
		pass
