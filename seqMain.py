#!/usr/bin/env python

"""
The main module for Event Trigger Identification via Deep Learning
"""
from __future__ import division

from os.path import join
from argparse import ArgumentParser
from pickle import load
from collections import defaultdict, Counter

from annotation import NIL_LABEL, EventMap
from basic import makeLogger
from main import evaluatePredictions
from ml.util import flatten
from basic import predictClasses

import keras.layers as kl
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model, Model
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM, GRU
import numpy as n

def seqLabelsToMatrix(labels, mapping):
	"""
	Applies the name to matrix sequentially
	"""
	return n.array([mapping.indexToMatrix(s) for s in labels])

def flattenLabels(labels, eventMapping):
	"""
	Converts the sequences of labels in into a flat sequence of label names
	"""
	return flatten(labels)

def buildModel(numClasses, seqLen, vecDim, contextDim):
	"""
	Builds a BiDirectional LSTM model on top of word embeddings, and doc2vec
	"""
	hidden = 256
	dense = 256
	cl2 = .001
	drop = .5

	#NOTE use functional api plus repeat layer for context vector
	#do a merge operation to join them

	seqInput = kl.Input((seqLen, vecDim))
	ctxInput = kl.Input((contextDim,))

	ctx = kl.RepeatVector(seqLen)(ctxInput)

	combined = kl.merge([seqInput, ctx], mode="concat")

	conj = kl.TimeDistributed(kl.Dense(dense, activation="relu"))(combined)
	conj = kl.TimeDistributed(kl.Dropout(drop))(conj)

	#cnn = kl.Conv1D(dense, 2, W_regularizer=l2(cl2))(conj)
	#pool = kl.MaxPooling1D(2)(cnn)

	seq = kl.Bidirectional(kl.LSTM(hidden, return_sequences=True, W_regularizer=l2(cl2), U_regularizer=l2(cl2), dropout_W=drop, dropout_U=drop), merge_mode="concat")(conj)

	seq = kl.Bidirectional(kl.LSTM(hidden, return_sequences=True, W_regularizer=l2(cl2), U_regularizer=l2(cl2), dropout_W=drop, dropout_U=drop), merge_mode="concat")(seq)

	out = kl.TimeDistributed(kl.Dense(numClasses, activation="softmax"))(seq)

	"""
	model = Sequential()
	#model.add(Embedding(vocabSize, len(initWeights[0]), weights=initWeights))
	#model.add(Embedding(vocabSize, len(initWeights[0])))
	#model.add(Embedding(vocabSize, 100))
	#model.add(Bidirectional(LSTM(hidden, return_sequences=True), merge_mode="concat"))
	model.add(Bidirectional(LSTM(hidden, return_sequences=True), input_shape=(seqLen, vecDim), merge_mode="concat"))
	#model.add(GRU(hidden, return_sequences=True, input_shape=(seqLen, vecDim)))
	model.add(TimeDistributed(Dense(numClasses)))
	model.add(TimeDistributed(Activation("softmax")))
	"""

	model = Model(input=[seqInput,ctxInput], output=out)

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	#print(model.summary())

	return model

def initWeights(wordIndex):
	path = "/home/walker/Data/vectors/glove/glove.6B.300d.txt"
	model = {}
	size = 0
	
	#parse the glove file
	for line in open(path):
		values = line.split(" ")
		word = values[0]
		vec = n.asarray(values[1:], dtype="float32")
		size = len(vec)

		model[word] = vec

	matrix = [None]*len(model)
	matrix[0] = n.zeros(size)

	for word, vec in model.items():
		matrix[wordIndex.index(word)] = vec

	return n.array(matrix)

def loadDir(directory):
	def l(name):
		return n.load(join(directory, name))

	results = {}

	results["train_x"] = [l("training_left.p"), l("training_right.p")]
	results["dev_x"] = [l("dev_left.p"), l("dev_right.p")]
	results["test_x"] = [l("test_left.p"), l("test_right.p")]

	results["train_y"] = l("training_labels.p")
	results["dev_y"] = l("dev_labels.p")
	results["test_y"] = l("test_labels.p")

	other = load(open(join(directory, "info.p")))
	results.update(other)

	return results

def main(args):
	"""
	Runs and evaluates the model
	"""
	print("Reading the data")
	dataDict = loadDir(args.f)

	trainData = dataDict["train_x"]
	devData = dataDict["dev_x"]
	testData = dataDict["test_x"]

	rawTrainingLabels = dataDict["train_y"] 
	rawDevLabels = dataDict["dev_y"] 
	rawTestingLabels = dataDict["test_y"] 
	
	#wordIndex = dataDict["word_index"]

	#make the event map
	eventMap = load(open(args.m))

	#TODO remove
	print(Counter(eventMap.toNames(flatten(rawTrainingLabels))).items())

	trainingLabels = seqLabelsToMatrix(rawTrainingLabels, eventMap)
	devLabels = seqLabelsToMatrix(rawDevLabels, eventMap)
	testingLabels = seqLabelsToMatrix(rawTestingLabels, eventMap)

	(samples, seqLen, dim) = trainData[0].shape
	(_, ctxDim) = trainData[1].shape

	print("#instances: {}, vector length: {}".format(samples, dim))
	#print("#instances: {}".format(len(trainLeftData)))
	
	print("Building the model")
	
	#get the model
	model = buildModel(len(eventMap), seqLen, dim, ctxDim)

	print("Training the model")
	
	#train the model

	#make the logger
	logger = makeLogger(args.o, eventMap)

	#TODO remove
	"""
	trainLeftData = n.array( [ [1,2,3], [1,5] ] )
	trainingLabels = n.array(seqNamesToMatrix( [ ["NIL", "NIL", "Attack"], ["NIL", "Attack"] ], eventMap ))
	devLeftData = trainLeftData
	devLabels = trainingLabels
	"""

	#model.fit(n.asarray(trainLeftData), n.asarray(trainingLabels), nb_epoch=args.e, batch_size=args.b, validation_data=(devLeftData, devLabels), class_weight=weights, callbacks=[logger])
	model.fit(trainData, trainingLabels, nb_epoch=args.e, batch_size=args.b, validation_data=(devData, devLabels), callbacks=[logger])

	#get the best model
	best = logger.best()

	print("Best Model round: {} val: {}".format(logger.bestModel, logger.bestScore))

	print("Make Predictions")
	#make predictions
	trainPred = predictClasses(best, trainData, args.b)
	devPred = predictClasses(best, devData, args.b)

	print("\nEvalutation")
	#evaluate the model

	print("-----Training Scores-----")
	evaluatePredictions(flatten(trainPred), eventMap.toNames(flatten(rawTrainingLabels)), eventMap)

	print("\n-----Dev Scores------")
	evaluatePredictions(flatten(devPred), eventMap.toNames(flatten(rawDevLabels)), eventMap)

	if args.t:
		testPred = best.predict_classes(testData, batch_size=args.b)
		print("\n\n-----Test Scores------")
		evaluatePredictions(flatten(testPred), eventMap.toNames(flatten(rawTestingLabels)), eventMap)

if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("-e", default=25, type=int, help="Number of epochs")
	parser.add_argument("-b", default=512, type=int, help="Batch size")
	parser.add_argument("-f", required=True, help="The pickle file with in the input data")
	parser.add_argument("-m", default="data/entity_event_map.p", help="The pickle file with the event map")
	parser.add_argument("-t", action="store_true", help="Evaluate on the testing set")
	parser.add_argument("-o", default="", help="The output directory for logging models and error rates")
		
	main(parser.parse_args())
