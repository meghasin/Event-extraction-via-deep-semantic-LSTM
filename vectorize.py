#!/usr/bin/env python

"""
A module for making vector representations of words, sentences etc
"""
from collections import defaultdict
from pickle import load
from itertools import groupby

from gensim.models.word2vec import KeyedVectors
from numpy import array, concatenate, zeros, asarray

from ner import NERIndex
from annotation import makeEntityMap, getKey

class Converter(object):
	"""
	This just defines an interface for code to convert words into vectors
	"""
	def convert(self, instance):
		"""
		Return a numpy array that represents some part (or all) of the instance
		"""
		raise Exception("Not implemented")

	def convertPhrase(self, token, sentence, docId, anchorId=0):
		pass

	def emptyVector(self):
		pass

	def __repr__(self):
		raise Exception("Not implemented")

def vectorize(instances, converters):
	"""
	Creates a matrix by vectorizing each instance with each converter.
	Each vector for an instance in concatenated together
	"""
	results = []

	#convert each instance
	for instance in instances:

		#vectorize it
		results.append(concatenate([c.convert(instance) for c in converters]))

	return array(results, "float32")

def loadW2V(fileName):
	"""
	Loads the word 2 vec model
	"""
	return KeyedVectors.load_word2vec_format(fileName, binary=True)

def loadGlove(fileName):
	"""
	Loads the glove
	"""
	model = {}

	with open(fileName) as inFile:
		
		#parse the glove file
		for line in inFile:
			values = line.split(" ")
			word = values[0]
			vec = asarray(values[1:], dtype="float32")

			model[word] = vec

	return model

def groupPhrases(index, window, sentence):
	for i in range(index - window, index + window + 1):
		if sentence.tokenExists(i):
			yield [sentence.tokenById(i)]
		else:
			yield []

def oldgroupPhrases(index, window, sentence):
	"""
	Group tokens if they are proper nouns
	"""
	tokens = sentence.tokens()

	#get the right hand context
	right = groupHalf(tokens[index:], window)

	#get the left hand context
	revLeft = groupHalf(reversed(tokens[:index+1]), window)

	left = list(reversed([list(reversed(l)) for l in revLeft]))
	
	#remove the center word from both halves of the window, we don't need the redundency
	rightHead = right[0]
	rightWindow = right[1:]

	leftHead = left[-1]
	leftWindow = left[:-1]

	#combine both centers if necessary
	headPhrase = sorted(list(set(rightHead).union(set(leftHead))))

	return leftWindow + [headPhrase] + rightWindow

properNoun = "PPN"

def groupKey(t):
	return properNoun if t.isProperNoun() else str(t.id)

def groupHalf(tokenStream, targetCount):
	"""
	Group up the tokens based on proper nouns
	"""
	results = []
	count = 0

	#group up the tokens by if they are 
	for key, phrase in groupby(tokenStream, key=groupKey):

		phrase = list(phrase)

		#if the phrase is proper noun phrase add it all
		if key == properNoun:

			phrases = [phrase]

		#else add all the words individually
		else:
			phrases = [[w] for w in phrase]

		#add each phrase or part one at a time
		for part in phrases:
			results.append(part)
			count += 1

			#if we found enough phrases return them
			if count == targetCount + 1:
				return results

	#pad out the remaining
	return results + ([ [] ] * ((targetCount +1) - len(results)))

class WindowFeats(Converter):
	"""
	Applies other converters to a window of tokens around the instance
	NOTE: Requires converters to have two additional methods: emptyVector and convertPhrase
	"""
	def __init__(self, converters, window):
		self.window = window
		self.converters = converters

	def convert(self, instance):
		"""
		Returns a matrix for the vectorized tokens around an instance
		"""
		index = instance.token.id
		sent = instance.sentence
		result = []

		#for each token in the window, look up the associtated vector
		for phrase in groupPhrases(index, self.window, sent):
			
			#if the token exists, get the token then lookup the vector
			if phrase:
				vectors = [c.convertPhrase(phrase, sent, instance.doc.id, index) for c in self.converters]

			#otherwise return an empty vector
			else:
				vectors = [c.emptyVector() for c in self.converters]
		
			result.append(concatenate(vectors))

		return array(result)

	def __repr__(self):
		return "Window Feats: {}".format(" ".join(map(str, self.converters)))

class Word2VecFeats(Converter):
	"""
	Applies word2vec to the instance token
	"""
	
	def __init__(self, model, window=0):
		"""
		Loads the model from the given file
		"""
		self.window = window
		self.model = model

	def convert(self, instance, concat=True):
		"""
		Applies word2vec to the instances token
		"""
		index = instance.token.id
		sent = instance.sentence
		result = []

		#for each token in the window, look up the associtated vector
		for i in range(index - self.window, index + self.window + 1):
			
			#if the token exists, get the token then lookup the vector
			if sent.tokenExists(i):
				result.append(self.convertPhrase([sent.tokenById(i)], sent, instance.doc.id))

			#otherwise return an empty vector
			else:
				result.append(self.emptyVector())	
		
		if concat:
			return concatenate(result)
		else:
			return array(result)

	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Returns a vector for the given token's word
		"""
		#if a token is given try to lookup an appropriate vector	
		if len(phrase) == 1:

			#NOTE not generic
			key = getKey(phrase[0], self.model)

			#if a vector was found use it
			if key:
				result = array(self.model[key])

			else:
				result = self.emptyVector()

		#look up the phrase
		elif len(phrase) > 1:
			key = "_".join([t.word for t in phrase])

			"""
			#TODO Remove
			found = key in self.model
			partsFound = any(t.word in self.model for t in phrase)
			allPPN = all(t.isProperNoun() for t in phrase)
			atLeast = any(t.isProperNoun() for t in phrase)

			#if not found and partsFound:
			if not allPPN and atLeast:
				ner = " ".join([t.ner for t in phrase])
				pos = " ".join([t.pos for t in phrase])
				ppn = " ".join([str(t.isProperNoun()) for t in phrase])
				ids = " ".join([str(t.id) for t in phrase])
				keys = " ".join([groupKey(t) for t in phrase])
				try:
					print("more: '{}', '{}', '{}', '{}', '{}', '{}', '{}'".format(key, found, ner, pos, ppn, ids, keys))
				except:
					print("problem")
			"""

			if key in self.model:
				result = array(self.model[key])
			else:
				result = self.emptyVector()
				
		else:
			result = self.emptyVector()

		return result

	def emptyVector(self):
		"""
		Returns a vector of zeros the same length as the w2v dimension
		"""
		#really hacky but I don't know how to get the size of the embeddings
		return zeros(len(self.model["dog"]))

	def __repr__(self):
		return "Word2Vec"

class Doc2VecFeats(Converter):
	"""
	Applies the Doc2Vec features to each document
	"""
	def __init__(self, fileName):	
		"""
		Loads the doc2vec model
		"""
		self.fileName = fileName
		self.model = self.load(fileName)

	def load(self, fileName):
		"""
		Loads the doc2vec model
		"""
		results = {}
		
		#read the embeddings for each document
		for line in open(fileName):
			docId, rawEmbedding = line.strip().split(": ")
			results[docId] = array(map(float,rawEmbedding.split(", ")))

		return results

	def convert(self, instance):
		"""
		Applies the doc2vec model to the document
		"""
		return self.model[instance.doc.id]
	
	def __repr__(self):
		return "Doc2Vec: {}".format(self.fileName)

class Sentence2VecFeats(Converter):
	"""
	Applies the Doc2Vec (sentence level) features to each sentence in each doc
	"""
	def __init__(self, fileName):	
		"""
		Loads the doc2vec model
		"""
		self.fileName = fileName
		self.model = self.load(fileName)

	def load(self, fileName):
		"""
		Loads the doc2vec model
		"""
		results = defaultdict(dict)
		
		#read the embeddings for each document
		for line in open(fileName):
			idPairs, rawEmbedding = line.strip().split(": ")
			docId, sentId = idPairs.split(",")
			results[docId][int(sentId)] = array(map(float,rawEmbedding.split(", ")))

		return results

	def convert(self, instance):
		"""
		Applies the doc2vec model to the document
		"""
		return self.model[instance.doc.id][instance.sentence.id]
	
	def __repr__(self):
		return "Sentence2Vec: {}".format(self.fileName)

class NERFeats(Converter):
	"""
	Makes Sparse NER Features
	"""

	def __init__(self, nerMapFile):
		"""
		Initialize the NER features with a mapping of NER types to indexes
		"""
		self.index = load(open(nerMapFile))

	def convert(self, instance):
		"""
		Makes a sparse vector for the instance
		"""
		return self.convertPhrase([instance.token], instance.sentence, instance.doc.id)

	def convertPhrase(self, token, sentence, docId, anchorId=0):
		"""
		Makes a sparse vector for the token
		"""
		vec = self.emptyVector()
		nerIndex = self.index.get(token.ner)

		if nerIndex != -1:
			vec[nerIndex] = 1.0

		return vec

	def emptyVector(self):
		"""
		Returns an empty vector of the correct size
		"""
		return zeros(len(self.index))

	def __repr__(self):
		return "NER"

class PositionFeats(Converter):
	"""
	Creates position based features for window based methods
	"""
	def __init__(self):
		self.dim = 2

	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Makes a sparse vector for the token
		"""
		vec = self.emptyVector()

		if phrase[0].id > anchorId:
			dist = min(t.id for t in phrase) - anchorId
		else:
			dist = max(t.id for t in phrase) - anchorId

		vec[0] = dist
		vec[1] = abs(dist)
		
		return vec

	def emptyVector(self):
		"""
		Returns an empty vector of the correct size
		"""
		return zeros(self.dim)

	def __repr__(self):
		return "Position"

class LemmaFeats(Converter):
	"""
	Makes a sparse vector
	"""

	def __init__(self, vocabFile):
		"""
		Initializes the convertor with a vocab file
		"""
		self.fileName = vocabFile
		self.index = {v.strip():i for i,v in enumerate(open(vocabFile))}

	def convert(self, instance):
		"""
		Makes a 'one hot' vector
		"""
		vec = zeros(len(self.index))

		wordIndex = self.index.get(instance.token.lemma, -1)

		if wordIndex != -1:
			vec[wordIndex] = 1.0

		return vec

	def __repr__(self):
		return "Lemmas: {}".format(self.fileName)

class EntityFeats(Converter):
	"""
	Uses gold entity information as features
	"""

	def __init__(self, entities):
		"""
		Uses entities to generate sparse features
		"""
		self.types = {t:i for i,t in enumerate(set([e.type for e in entities]))}
		self.mapping = makeEntityMap(entities)

	def convert(self, instance):
		"""
		Return a numpy array that represents some part (or all) of the instance
		"""
		return self.convertPhrase([instance.token], instance.sentence, instance.doc.id)

	#NOTE: does not work with phrases
	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Converts a phrase into a sparse entity vector
		"""
		key = (docId, phrase[0].sentenceId, phrase[0].id)
		vec = self.emptyVector()
	
		#look up the entity info
		ent = self.mapping.get(key, None)
	
		#if there is an entity, get its type
		if ent:
			index = self.types[ent.type]
			vec[index] = 1.0

		return vec

	def emptyVector(self):
		"""
		Returns an empty vector of the correct size
		"""
		return zeros(len(self.types))

	def __repr__(self):
		return "Sparse Entity Feats"

class SparsePOSFeats(Converter):
	"""
	Makes sparse POS Features
	"""
	def __init__(self, posTags):
		"""
		Makes pos features from the given tags
		"""
		self.mapping = {t:i for i,t in enumerate(posTags)}

	def convert(self, instance):
		"""
		Return a numpy array that represents some part (or all) of the instance
		"""
		return self.convertPhrase([instance.token], instance.sentence, instance.doc.id)

	#NOTE: doesn't work for phrases
	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Converts a phrase into a sparse entity vector
		"""
		vec = self.emptyVector()

		#look up the pos mapping
		index = self.mapping.get(phrase[0].pos, None)

		if index is not None:
			vec[index] = 1.0
		
		return vec

	def emptyVector(self):
		"""
		Returns an empty vector of the correct size
		"""
		return zeros(len(self.mapping))

	def __repr__(self):
		return "Sparse POS Feats"

class SparseDependencyFeats(Converter):
	"""
	Makes sparse dependency Features
	"""
	def __init__(self, depTags):
		"""
		Makes dependency features from the given tags
		"""
		self.mapping = {t:i for i,t in enumerate(depTags)}

	def convert(self, instance):
		"""
		Return a numpy array that represents some part (or all) of the instance
		"""
		return self.convertPhrase([instance.token], instance.sentence, instance.doc.id)

	#NOTE: doesn't work for phrases
	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Converts a phrase into a sparse entity vector
		"""
		vec = self.emptyVector()
		
		#look up the token's type
		dep = sentence.getTokenType(phrase[0])
	
		#if the token has a type
		if dep:
			
			#look up the pos mapping
			vec[self.mapping[dep]] = 1.0
		
		return vec

	def emptyVector(self):
		"""
		Returns an empty vector of the correct size
		"""
		return zeros(len(self.mapping))

	def __repr__(self):
		return "Sparse POS Feats"

class SparseDocTypeFeats(Converter):
	"""
	Makes sparse features based on the document type
	"""

	def __init__(self, docTypeFile):
		"""
		Creates a mapping from doc type to sparse features
		"""
		docTypes = [l.strip().split(" ") for l in open(docTypeFile)]
		self.mapping = {docId:t for t, docId in docTypes}
		self.index = {t:i for i,t in enumerate(set(self.mapping.values()))}

	def convert(self, instance):
		"""
		Return a numpy array that represents some part (or all) of the instance
		"""
		return self.convertPhrase([instance.token], instance.sentence, instance.doc.id)

	#NOTE: doesn't work for phrases
	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Converts a phrase into a sparse entity vector
		"""
		vec = self.emptyVector()
		
		#look up the doc type mapping
		vec[self.index[self.mapping[docId]]] = 1.0
		
		return vec

	def emptyVector(self):
		"""
		Returns an empty vector of the correct size
		"""
		return zeros(len(self.mapping))

	def __repr__(self):
		return "Sparse Doc Type Feats"

class EmbeddingFeats(Converter):
	"""
	Converts phrase into indexes to use in an embedding later
	"""
	def __init__(self, index):
		"""
		Creates index features based on the word index
		"""
		self.index = index 
		self.padding = 0
		self.outOfVocab = 1

	def convert(self, instance):
		"""
		Return a numpy array that represents some part (or all) of the instance
		"""
		return self.convertPhrase([instance.token], instance.sentence, instance.doc.id)

	#NOTE: doesn't work for phrases
	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Converts a phrase into a vector of indexes
		"""
		pass

	def emptyVector(self):
		"""
		Returns an empty vector of the correct size
		"""
		return zeros(1, dtype="int32")

	def __repr__(self):
		return "Embedding Feats"

class WordEmbeddingFeats(EmbeddingFeats):
	"""
	Makes word indexes for an embedding
	"""
	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Converts a phrase into word index vector
		"""
		vec = self.emptyVector()
	
		#look up the doc type mapping
		#vec[0] = self.index.get(phrase[0].word, self.outOfVocab)
		vec[0] = self.index.index(phrase[0])
		
		return vec

class DistanceEmbeddingFeats(EmbeddingFeats):
	"""
	Returns a distance vector
	"""
	def __init__(self):
		pass

	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Converts a phrase into word index vector
		"""
		vec = self.emptyVector()

		#look up the doc type mapping
		vec[0] = abs(phrase[0].id - anchorId)

		#TODO remove
		print("distance: id {}, anch {}, dist {}".format(phrase[0].id, anchorId, vec[0]))

		return vec

class EntityEmbeddingFeats(EmbeddingFeats):
	"""
	Return entity indexes
	"""
	def __init__(self, index, entities):
		"""
		Use the index and entity information for the embedding
		"""
		self.index = index
		self.mapping = makeEntityMap(entities)

	def convertPhrase(self, phrase, sentence, docId, anchorId=0):
		"""
		Converts a phrase into a sparse entity vector
		"""
		key = (docId, phrase[0].sentenceId, phrase[0].id)
		vec = self.emptyVector()
	
		#look up the entity info
		ent = self.mapping.get(key, None)
	
		#if there is an entity, get its type
		if ent:
			vec[0] = self.index.index[ent.type]

		return vec
