#!/usr/bin/env python

"""
A module for handling event annotation and annotated document reading
"""

import csv
from os import listdir
from os.path import join

from keras.utils import np_utils
import numpy as n

from ml.dependency import parseDocument, parseId
import config as c

NIL_LABEL = "NIL"
ACTUAL = "Actual"
GENERIC = "Generic"

class Event(object):

	def __init__(self, id, docId, type, sentenceId, tokenId):
		"""
		Initialize the Event
		"""
		self.id = id
		self.docId = docId
		self.type = type
		self.sentenceId = sentenceId
		self.tokenId = tokenId

	def isNil(self):
		return self.type == NIL_LABEL

	def __repr__(self):
		return "{}: {}, {}-{}".format(self.id, self.type, self.sentenceId, self.tokenId)

class Entity(object):

	def __init__(self, id, docId, type, sentenceId, tokenStart, tokenEnd):
		self.id = id
		self.docId = docId
		self.type = type
		self.sentenceId = sentenceId
		self.startId = tokenStart
		self.endId = tokenEnd

	def isFirstToken(self, token):
		"""
		Returns true if the token matches the first token included in the entity
		"""
		return token.sentenceId == self.sentenceId and token.id == startId

	def __repr__(self):
		return "Ent:{} {} {} {} {}".format(self.id, self.type, self.sentenceId, self.startId, self.endId)

def readEvents(fileName):
	"""
	Reads and returns events listed in the file
	"""
	results = []
	skip = True

	if fileName:
		
		#read each line, read in the event info
		for line in csv.reader(open(fileName)):
		
			#skip the first line of the file
			if not skip:
				(id, doc, type, sentence, token) = line

				results.append(Event(id, doc, type, int(sentence), int(token)))

			skip = False

	return results

def readEntities(fileName):
	"""
	Reads and returns entities listed in the file
	"""
	results = []
	skip = True

	#read each line, read in the event info
	for line in csv.reader(open(fileName)):
	
		#skip the first line of the file
		if not skip:
			(id, doc, type, context, sentence, start, end) = line

			results.append(Entity(id, doc, type, int(sentence), int(start), int(end)))

		skip = False

	return results


def collectDocIds(events):
	"""
	Collects the doc ids from all the list of events
	"""
	results = set()

	for event in events:
		results.add(event.docId)

	return results

def readDocs(path, events):
	"""
	Reads the annotated documents at the given path
	"""
	results = []

	#determine the files needed
	docs = collectDocIds(events)

	#read the files at the given path
	for name in listdir(path):
		if parseId(name) in docs:
			results.append(parseDocument(join(path, name)))

	return results

class Instance(object):
	"""
	A data point, a single token to be classified with its context
	"""

	def __init__(self, token, sentence, doc, event):
		"""
		Create an instance with its context
		"""
		self.token = token
		self.sentence = sentence
		self.doc = doc
		self.event = event

	def isNil(self):
		return self.event.isNil()

	def isRealis(self):
		return self.event.type in {ACTUAL, GENERIC}

	def __repr__(self):
		return "T{} in S{}-{}".format(self.token.id, self.sentence.id, self.doc.id)

class SequenceTag(object):
	"""
	Has information about the sentence
	"""
	def __init__(self, sentenceId, docId, chunkIndex):
		self.docId = docId
		self.sentenceId = sentenceId
		self.chunkIndex = chunkIndex

	def __repr__(self):
		return "Seq c:{} s:{} d:{}".format(self.chunkIndex, self.sentenceId, self.docId)

class SequenceInstance(object):
	"""
	Represents a sentence - a sequence of tokens to make a predictions for
	doc - the document
	sentence - the sentence
	instances - list of per token instances
	chunkIndex - which chunk of the sentence this is
	"""
	def __init__(self, sentence, doc, instances, chunkIndex=0):
		self.doc = doc
		self.sentence = sentence
		self.instances = instances
		self.chunkIndex = chunkIndex

	def toTag(self):
		return SequenceTag(self.sentence.id, self.doc.id, self.chunkIndex)

	def changeInstances(self, newInstances, chunk):
		return SequenceInstance(self.sentence, self.doc, newInstances, chunk)

def makeEntityMap(entities):
	"""
	Returns a map for (doc, sentence, token) -> entity
	"""
	result = {}

	#add a mapping for each entity and each token it contains
	for ent in entities:
		for token in range(ent.startId, ent.endId+1):
			result[(ent.docId, ent.sentenceId, token)] = ent

	return result

def makeEventMap(events):
	"""
	Returns a map for (doc, sentence, token) -> event
	"""
	return {(e.docId, e.sentenceId, e.tokenId):e for e in events}

def createInstances(docs, events, includeAll=False):
	"""
	Returns a list of instances produced from the given set of documents
	"""
	results = []
	labels = []

	eventMap = makeEventMap(events)

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
					event = eventMap.get(key, None)

					#make a NIL event if necessary
					if event is None:
						event = Event("NIL-{}-{}-{}".format(doc.id, sentence.id, token.id), doc.id, NIL_LABEL, sentence.id, token.id)

					results.append(Instance(token, sentence, doc, event))
					
					#add the label
					labels.append(event.type)

	return results, labels

def segmentSequences(sequence, labels):
	"""
	Breaks up sequences according to the max length
	"""
	if len(labels) <= c.maxLen:
		return [(sequence, labels)]
	else:
		leftInst = sequence.instances[:c.maxLen]
		rightInst = sequence.instances[c.maxLen:]
		leftLabels = labels[:c.maxLen]
		rightLabels = labels[c.maxLen:]

		left = sequence.changeInstances(leftInst, sequence.chunkIndex)
		right = sequence.changeInstances(rightInst, sequence.chunkIndex+1)

		return [(left, leftLabels)] + segmentSequences(right, rightLabels)

def createSequenceInstances(docs, events, entities, mapping, bio=False):
	"""
	Returns a list of lists - one per sentence in the corpus
	"""
	results = []
	labels = []

	eventMap = makeEventMap(events)
	entityMap = makeEntityMap(entities)

	#create instances for the tokens in each document
	for doc in docs:

		#for each sentence, make instances out of each token
		for sentence in doc.sentences:

			sentInst = []
			sentLabels = []

			#for each token, if it is a noun, pronoun, verb, or adj make it an instance
			for token in sentence.tokens():

				#see if this token has an annotation
				key = (doc.id, sentence.id, token.id) 
				event = eventMap.get(key, None)
				entity = entityMap.get(key, None)

				sentInst.append(Instance(token, sentence, doc, event))
				
				#add the label
				if event is None and entity is None:
					sentLabels.append(NIL_LABEL)
				
				elif event:
					sentLabels.append(event.type)

				else:
					#use BIO tags
					if bio:
						label = "B-{}" if entity.firstToken(token) else "I-{}"
						sentLabels.append(label.format(entity.type))

					else:
						sentLabels.append(entity.type)
	
			#break the sentences into same length chunks
			for seq, seqLabels in segmentSequences(SequenceInstance(sentence, doc, sentInst), mapping.toIndex(sentLabels)):
				results.append(seq)
				labels.append(seqLabels)

	return results, labels

class EventMap(object):
	"""
	Maps between event labels and their index
	"""
	def __init__(self, labels):
		"""
		Initalize the map with all the data's labels
		"""
		self.index = {l:i for i,l in enumerate(set(labels))}
		self.revIndex = {i:l for l,i in self.index.items()}

	def toIndex(self, labels):
		"""
		Turns a list of event names into a list of event indexes
		"""
		return [self.index[l] for l in labels]

	def indexToMatrix(self, labels):
		"""
		Converts the list of label indexes to a matrix
		"""
		return np_utils.to_categorical(labels, len(self))

	def namesToMatrix(self, labels):
		"""
		Converts the list of labels (name) into a matrix
		"""
		return np_utils.to_categorical(self.toIndex(labels), len(self))

	def toNames(self, vector):
		"""
		Converts all the indexes to names
		"""
		results = []
		
		for i in vector:
			results.append(self.revIndex[i.item()])

		return results

	def matrixToIndex(self, matrix):
		"""
		Converts the matrix to an index vector
		"""
		results = []
		
		#look for the "one hot" value
		for row in matrix:
			results.append([i for i,v in enumerate(row) if v][0])

		return n.array(results)

	def matrixToNames(self, matrix):
		"""
		Converts a matrix into a list of class names
		"""
		return self.toNames(self.matrixToIndex(matrix))

	def allLabels(self):
		"""
		Returns all the labels in the map
		"""
		return self.index.keys()

	def eventLabels(self):
		"""
		Returns all the labels without the Nil type
		"""
		return [c for c in self.allLabels() if c != NIL_LABEL]

	def eventIndexes(self):
		"""
		Returns the indexes of all the events minus the NIL type
		"""
		return [i for i in self.index.values() if i != self.nilIndex()]

	def nilIndex(self):
		"""
		Returns the index of the NIL type
		"""
		return self.index[NIL_LABEL]

	def __len__(self):
		return len(self.index)

def getKey(token, vocab):
	"""
	Determines the best key for the token
	"""
	lower = token.word.lower()

	if token.word in vocab:
		return token.word

	elif lower in vocab:
		return lower

	elif token.isNER():
		try:
			print("NER: {}-{}".format(token.word, token.ner))
		except:
			pass

		return token.ner

	else:
		return None


class TokenIndex(object):
	"""
	Maps token to indexes
	"""
	def __init__(self, start=1, vocab=[]):
		"""
		Initalizes the index with a starting value and vocab
		"""
		self.mapping = {w:i+start for i,w in enumerate(vocab)}
		self.outOfVocab = 0
		self.maxIndex = len(vocab)+1 

	def index(self, token):
		"""
		Returns the index of the token or returns zero
		"""
		word = token.word
		lower = word.lower()

		#first try the word itself
		if word in self.mapping:
			return self.mapping[word]
	
		#try the lower case, version
		elif lower in self.mapping:
			return self.mapping[lower]

		#try NER
		elif token.ner in self.mapping:
			return self.mapping[token.ner]

		else:
			return self.outOfVocab

	def key(self, token, vocab):
		"""
		Determines which key should be used
		"""
		return getKey(token, vocab)

	def updateIndex(self, token, vocab):
		"""
		Updates if necessary and returns the token index
		"""
		tokenKey = self.key(token, vocab)
		tokenIndex = self.index(token)

		if tokenKey and not tokenIndex:
			self.mapping[tokenKey] = self.maxIndex
			self.maxIndex += 1

	def __len__(self):
		return len(self.mapping)
