#!/usr/bin/env python

"""
Trains an NER embedding to be used in the event task
"""

from os import listdir
from os.path import join
from pickle import dump
from argparse import ArgumentParser

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation

from ml.util import flatten
from ml.dependency import loadDocuments
import config as c

NO_NER = "NO_NER"

def docToSequences(document, maxLen):
	"""
	Converts the document into a list of sequences of NER
	"""
	results = []

	#for each sentence, convert it into a sequence of NER tags
	for sentence in document.sentences:
		
		#get the ner tags
		ner = [t.ner for t in sentence.tokens()]
		size = len(ner)

		#make sure it conforms to the correct length
		if size < maxLen:
			ner = ner + ([NO_NER] * (maxLen - size))
		
		elif size > maxLen:
			ner = ner[:maxLen]

		results.append(ner)

	return results

def buildSequences(stanfordDir, maxLen):
	"""
	Builds sequences out of sentences in documents that are stanford annotated
	"""
	results = []

	#for each document, extract all the sequences of ner tags
	for doc in loadDocuments(stanfordDir):
		for seq in docToSequences(doc, maxLen):	
			results.append(seq)

	return results

def buildIndex(sequences):
	"""
	Builds an index that maps NER tags to indexes
	"""
	return NERIndex(set(flatten(sequences)))

class NERIndex(object):
	"""
	A mapping from NER tag to index
	"""

	def __init__(self, tags):
		"""
		Initialize the index
		"""
		self.index = {t:i for i,t in enumerate(sorted(tags))}

	def sequencesToIndex(self, sequences):
		"""
		Converts the list of ner sequences to a list of index sequences
		"""
		return [self.toIndex(s) for s in sequences]

	def toIndex(self, nerSeq):
		"""
		Converts the ner sequence into a list of indexes
		"""
		return [self.index[n] for n in nerSeq]

	def get(self, nerTag):
		"""
		Returns the index of the NER tag or -1
		"""
		return self.index.get(nerTag, -1)

	def __len__(self):
		return len(self.index)

def trainEmbeddings(sequences, embeddingDim, numTypes, maxLen):
	"""
	Trains an embedding for NER over the sequences
	"""
	#define the model
	model = Sequential()
	model.add(Embedding(numTypes, embeddingDim, maxLen))
	model.add(TimeDistributed(Activation("Softmax")))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	#train the model
	model

def main(args):
	"""
	Builds an NER embeddings
	"""
	#build the sequences
	sequences = buildSequences(c.dataPath, args.m)

	#build the index
	index = buildIndex(sequences)

	#vectorize the sequences
	indexSeq = index.sequencesToIndex(sequences)

	#train embedding
	#weights = trainEmbeddings(sequences, args.d, len(index), args.m)

	#save index
	with open(join(args.out, "nerIndex.p"), "w") as indexOut:
		dump(index, indexOut)

	#save weights
	#with open(join(args.out, "nerWeights.p"), "w") as weightsOut:
		#dump(weights, weightsOut)

if __name__ == "__main__":
	
	parser = ArgumentParser()

	parser.add_argument("out", help="The output directory")
	parser.add_argument("-m", type=int, default=50, help="The max sentence length")
	parser.add_argument("-d", type=int, default=1, help="The embedding dimension")

	main(parser.parse_args())
