#!/usr/bin/env python

"""
The main module for Event Trigger Identification via Deep Learning
"""
from __future__ import division

from argparse import ArgumentParser
from pickle import load
from collections import defaultdict
from os.path import join

from basic import makeLogger, microF1, predictClasses, weightClasses
from main import evaluatePredictions
from vectorize import loadW2V

from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model, Model
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.constraints import MaxNorm

import numpy as n

from ml.util import mkdir

wordEmbName = "word_emb"
distEmbName = "dist_emb"
entEmbName = "ent_emb"
cnnModelName = "cnn_model"
cnnName = "cnn_{}"
cnnModelNameAlt= "model_2"

#translation from default names to assigned names
translation = {"embedding_1":wordEmbName,
"embedding_2":entEmbName,
distEmbName:distEmbName,
"convolution1d_1":cnnName.format(2),
"convolution1d_2":cnnName.format(3),
"convolution1d_3":cnnName.format(4),
"convolution1d_4":cnnName.format(5)}

def loadData(path):
	"""
	Loads the data from either format
	"""
	def l(name):
		return n.load(join(path, name))

	#if the path is a directory, load all the files in it
	results = {}

	results["train_x"] = l("training_left.p")
	results["dev_x"] = l("dev_left.p")
	results["test_x"] = l("test_left.p")

	results["train_y"] = l("training_labels.p")
	results["dev_y"] = l("dev_labels.p")
	results["test_y"] = l("test_labels.p")

	other = load(open(join(path, "info.p")))
	results.update(other)

	return results

def setupEmbeddings(data):
	"""
	Converts the first matrix into multiple series of embedding indexes
	"""
	#split up the window data into as many data sets as there are columns 
	samples, seqLen, columns = data.shape

	splitData = [m.reshape(samples, seqLen) for m in n.array_split(data, columns, 2)]

	return splitData

def makeEmbeddingWeights(w2vModel, wordIndex):
	"""
	Creates the intial embedding weights
	"""
	maxIndex = max(wordIndex.mapping.values())

	#TODO fix hack
	#dim = len(w2vModel["dog"])
	dim = 300

	matrix = n.zeros((maxIndex+1, dim))

	#for each word, index combo build a row of the matrix
	for word, index in wordIndex.mapping.items():
		if word in w2vModel:
			matrix[index] = w2vModel[word]
		else:
			matrix[index] = n.random.uniform(-.05, .05, dim)

	return matrix

def tripleEmbedding(initWeights, seqLen, entDim, distDim, maxDist, numEnts, contextDim=0):
	"""
	Makes an embedding for words, position and entities
	"""
	wordInput = Input(shape=(seqLen,))
	distInput = Input(shape=(seqLen,))
	entInput = Input(shape=(seqLen,))
	contextInput = Input(shape=(contextDim,))

	wordW = initWeights.get(wordEmbName, None)
	distW = initWeights.get(distEmbName, None)
	entW = initWeights.get(entEmbName, None)

	(vocab, dim) = wordW[0].shape

	wordEmb = Embedding(vocab, dim, weights=wordW, input_length=seqLen, name=wordEmbName)(wordInput)
	distEmb = Embedding(maxDist, distDim, weights=distW, input_length=seqLen, name=distEmbName, init="lecun_uniform")(distInput)
	entEmb = Embedding(numEnts, entDim, weights=entW, input_length=seqLen, name=entEmbName)(entInput)
	context = RepeatVector(seqLen)(contextInput)

	if contextDim:
		layers = [wordEmb, entEmb, distEmb, context]
		inputs = [wordInput, entInput, distInput, contextInput]
	else:
		layers = [wordEmb, entEmb, distEmb]
		inputs = [wordInput, entInput, distInput]

	out = merge(layers, mode="concat", concat_axis=-1)

	return Model(input=inputs, output=out)

def multiCNN(inputShape, numFilters, sizeRange, cl2, rnnSize=0, initWeights={}):
	"""
	Returns a 1d CNN layer that has multiple filter sizes
	"""
	convLayers = []

	#setup the input
	start = Input(shape=inputShape)

	#add all the CNN filters
	for size in sizeRange:
		name = cnnName.format(size)
		init = initWeights.get(name, None)

		if init is not None:
			print("Loading Weights for {}".format(name))

		x = Conv1D(numFilters, size, W_regularizer=l2(cl2), activation="relu", name=name, weights=init)(start)

		if rnnSize:
			x = MaxPooling1D(size)(x)
			x = Bidirectional(GRU(rnnSize, W_regularizer=l2(cl2), U_regularizer=l2(cl2)))(x)
		else:
			x = GlobalMaxPooling1D()(x)

		convLayers.append(x)

	#merge the outputs
	#model = Model(input=start, output=merge(convLayers, mode="concat", concat_axis=-1), name=cnnModelName)

	#model.summary()

	#return model
	return Model(input=start, output=merge(convLayers, mode="concat", concat_axis=-1), name=cnnModelName)

def buildCNNEmbModel(numClasses, seqLen, contextDim, initWeights, eventMap):
	"""
	Reproduces an CNN with embedding
	"""
	cnnHidden = 75
	cl2 = .000
	drop = .5
	convSize = [2,3,4,5]

	entDim = 20
	distDim = 2
	fullEmbeddingDim = 300 + entDim + distDim 

	#NOTE hard coded for dataset
	shape = (seqLen, fullEmbeddingDim)

	model= Sequential()

	init = {wordEmbName:[initWeights]}

	#NOTE hardcoded for dataset
	emb = tripleEmbedding(init, seqLen, entDim, distDim, 12, 13)
	cnn = multiCNN(shape, cnnHidden, convSize, cl2)

	model.add(emb)
	model.add(cnn)
	model.add(Dropout(drop))
	
	model.add(Dense(numClasses, W_constraint=MaxNorm(3.0)))
	model.add(Activation("softmax"))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	print(model.summary())

	return model

def buildCNNEmbOutput(numClasses, seqLen, contextDim, eventMap, fullModel):
	"""
	Builds a reproduction of the model without the classification layer
	"""
	cnnHidden = 75
	cl2 = .000
	drop = .5
	convSize = [2,3,4,5]

	entDim = 20
	distDim = 2
	fullEmbeddingDim = 300 + entDim + distDim 

	#NOTE hard coded for dataset
	shape = (seqLen, fullEmbeddingDim)

	model= Sequential()

	init = {}

	names = [wordEmbName, distEmbName, entEmbName] + [cnnName.format(c) for c in convSize]

	for name in names:

		layer = fullModel.get_layer(name)

		if layer is None:
			layer = fullModel.get_layer(cnnModelName).get_layer(name)

		init[name] = layer.get_weights()

	#NOTE hardcoded for dataset
	emb = tripleEmbedding(init, seqLen, entDim, distDim, 12, 13)
	cnn = multiCNN(shape, cnnHidden, convSize, cl2, 0, init)

	model.add(emb)
	model.add(cnn)
	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	model.summary()

	return model

def buildMultiEmbModel(numClasses, seqLen, contextDim, initWeights, eventMap):
	"""
	Does a RNN over a CNN model with context information
	"""
	cnnHidden = 150
	rnnHidden = 64
	cl2 = .000
	drop = .5
	convSize = [2,3,4,5]

	entDim = 20
	distDim = 2

	#NOTE hard coded for dataset
	#shape = (seqLen, 300 + entDim + distDim + contextDim)
	shape = (seqLen, 300 + entDim + distDim )

	model= Sequential()

	init = {wordEmbName:[initWeights]}

	#NOTE hardcoded for dataset
	emb = tripleEmbedding(init, seqLen, entDim, distDim, 12, 13, 0)

	#TODO remove
	print("embedding input {}".format(emb.input_shape))
	print("embedding output {}".format(emb.output_shape))
	print("next level shape {}".format(shape))

	cnn = multiCNN(shape, cnnHidden, convSize, cl2)

	model.add(emb)
	model.add(cnn)
	
	#model.add(Conv1D(512, 2))
	#model.add(MaxPooling1D(2))
	#model.add(Bidirectional(GRU(128)))

	simple = Input(shape=(contextDim,))
	level2 = Model(input=simple, output=simple)

	level3 = Sequential()

	level3.add(Merge([model, level2], mode="concat"))

	#level3.add(Dense(256))
	#level3.add(Activation("relu"))

	level3.add(Dropout(drop))
	
	level3.add(Dense(numClasses, W_constraint=MaxNorm(3.0)))
	level3.add(Activation("softmax"))

	level3.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy', microF1(eventMap)])

	print(level3.summary())

	return level3

def eventKey(event):
	return (event.docId, event.sentenceId, event.tokenId)

def padRealisSet(events, realis, realisMatrix):
	"""
	Extends a realis dataset to match the dimension of the events
	"""
	results = []
	realisIndex = {eventKey(r):i for i,r in enumerate(realis)}
	(_, dim) = realisMatrix.shape

	#for each event lookup the relevant realis data
	for event in events:
		
		index = realisIndex.get(eventKey(event), None)

		if index is not None:
			vector = realisMatrix[index]
		else:
			vector = n.zeros(dim)

		results.append(vector)

	return n.array(results)

def loadEvents(path):
	"""
	Loads event from the info file at the given path
	"""
	train = "train_events"
	dev = "dev_events"
	test = "test_events"
	names = [train, dev, test]

	data = load(open(join(path, "info.p")))

	return [data[n] for n in names]

def padRealis(eventsPath, realisPath, realisData):
	"""
	Realis is not specified for all the possible events, only those candidates
	that match
	"""
	#load event data
	events = loadEvents(eventsPath)

	#load realis data
	realis = loadEvents(realisPath)

	#extend each dataset
	return [padRealisSet(e,r,d) for e,r,d in zip(events, realis, realisData)]

def predictRealis(model, datasets):
	"""
	Predicts the hidden layer on each dataset
	"""
	return [model.predict(d) for d in datasets]

def writeRealis(datasets, names):
	"""
	Writes out all the datasets to the given locations
	"""
	for data, name in zip(datasets, names):
		
		print("Outshape {}".format(data.shape))
		n.save(name, data)

def main(args):
	"""
	Runs and evaluates the model
	"""
	#n.random.seed(13)
	n.random.seed(16)

	print("Reading the data")
	dataDict = loadData(args.f)

	useEmb = args.full

	if args.o:
		mkdir(args.o)

	#unpack the data
	trainData = setupEmbeddings(dataDict["train_x"])
	devData = setupEmbeddings(dataDict["dev_x"])
	testData = setupEmbeddings(dataDict["test_x"])
	
	rawTrainingLabels = dataDict["train_y"] 
	rawDevLabels = dataDict["dev_y"] 
	rawTestingLabels = dataDict["test_y"] 
	
	#make the event map
	eventMap = load(open(args.m))

	trainingLabels = eventMap.namesToMatrix(rawTrainingLabels)
	devLabels = eventMap.namesToMatrix(rawDevLabels)
	testingLabels = eventMap.namesToMatrix(rawTestingLabels)

	(samples, seqLen) = trainData[0].shape

	print(trainData[0].shape)

	(rightSamples, contextDim) = trainData[-1].shape

	print("#instances: {}, seq len: {}".format(samples, seqLen))
	print("right side {} {}".format(rightSamples, contextDim))
	print("labels shape {}".format(trainingLabels.shape))
	
	print("Building the model")
	
	#get the model
	w2vPath = "data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"
	indexPath = "data/word_index.p"

	#load the weights
	#maybe it was commented for run 13?
	w2v = loadW2V(w2vPath)
	#w2v = {}

	#load the index
	wordIndex = load(open(indexPath))

	#make the initial weights
	initWeights = makeEmbeddingWeights(w2v, wordIndex)

	if args.full:
		model = buildMultiEmbModel(len(eventMap), seqLen, contextDim, initWeights, eventMap)

	else:
		model = buildCNNEmbModel(len(eventMap), seqLen, contextDim, initWeights, eventMap)

	#train the model
	print("Training the model")
	
	#hard coding class weights...
	#weights = {0:1.0, 1:5.5}
	weights = {0:1.0, 1:6.0}
	#weights = {0:1.0, 1:9.0}

	#make the logger
	logger = makeLogger(args.o, eventMap)

	model.fit(trainData, trainingLabels, nb_epoch=args.e, batch_size=args.b, 
		validation_data=(devData, devLabels), 
		callbacks=[logger], class_weight=weights)

	#get the best model
	best = logger.best()

	print("Best Model round: {} val: {}".format(logger.bestModel, logger.bestScore))
	#print("F1 Best Model round: {} val: {}".format(sndLog.bestModel, sndLog.bestScore))

	print("Make Predictions")
	#make predictions
	trainPred = predictClasses(best, trainData, args.b)
	devPred = predictClasses(best, devData, args.b)

	print("\nEvalutation")
	#evaluate the model

	print("-----Training Scores-----")
	evaluatePredictions(trainPred, rawTrainingLabels, eventMap)

	print("\n-----Dev Scores------")
	evaluatePredictions(devPred, rawDevLabels, eventMap)

	if args.t:
		testPred = predictClasses(best, testData, args.b)
		print("\n\n-----Test Scores------")
		evaluatePredictions(testPred, rawTestingLabels, eventMap)

	#output the embedded layer
	if args.out and not args.full:

		realisOut = "realisOut"
		realisProbOut = "realisProbOut"

		makeNames = lambda p: [join(args.o, p, i) for i in ["training_pred", "dev_pred", "test_pred"]]

		outModel = buildCNNEmbOutput(len(eventMap), seqLen, contextDim, eventMap, best)

		#do realis layer prediction
		realis = predictRealis(outModel, [trainData, devData, testData])
		realisPaths = makeNames(realisOut)
		mkdir(join(args.o, realisOut))

		#do realis prob prediction
		realisProb = predictRealis(best, [trainData, devData, testData])
		realisProbPaths = makeNames(realisProbOut)
		mkdir(join(args.o, realisProbOut))
	
		writeRealis(padRealis(args.eventPath, args.f, realis), realisPaths)
		writeRealis(padRealis(args.eventPath, args.f, realisProb), realisProbPaths)

if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("-e", default=10, type=int, help="Number of epochs")
	parser.add_argument("-b", default=1024, type=int, help="Batch size")
	parser.add_argument("-f", required=True, help="The pickle file with in the input data")
	parser.add_argument("-m", default="data/realis_map.p", help="The pickle file with the event map")
	parser.add_argument("-t", action="store_true", help="Evaluate on the testing set")
	parser.add_argument("-o", default="", help="The output directory for logging models and error rates")
	parser.add_argument("-full", action="store_true", help="Use the full model")
	parser.add_argument("-out", action="store_true", help="Output the layer before the classifier layer")
	parser.add_argument("-eventPath", required=True, help="The path to the output")

	main(parser.parse_args())
