#!/usr/bin/env python

"""
The main module for Event Trigger Identification via Deep Learning
"""
from __future__ import division

from argparse import ArgumentParser
from pickle import load
from collections import defaultdict
from os.path import join

from basic import makeLogger, loadData, microF1, predictClasses, weightClasses
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
from realisMain import tripleEmbedding, multiCNN
import realisMain as rm

def setupEmbeddings(data, useBothHalves):
	"""
	Converts the first matrix into multiple series of embedding indexes
	"""
	window, context = data

	#split up the window data into as many data sets as there are columns 
	samples, seqLen, columns = window.shape

	#make each column its own matrix (vector) and include the context matrix as well
	splitData = [m.reshape(samples, seqLen) for m in n.array_split(window, columns, 2)]

	if useBothHalves:
		return splitData + [context]

	else:
		return splitData

def makeEmbeddingWeights(w2vModel, wordIndex):
	"""
	Creates the intial embedding weights
	"""
	maxIndex = max(wordIndex.mapping.values())

	#TODO fix hack
	dim = len(w2vModel["dog"])

	matrix = n.zeros((maxIndex+1, dim))

	#for each word, index combo build a row of the matrix
	for word, index in wordIndex.mapping.items():
		if word in w2vModel:
			matrix[index] = w2vModel[word]
		else:
			matrix[index] = n.random.uniform(-.05, .05, dim)

	return matrix

def seqCombine(layers):
	from keras.backend import batch_dot
	seq, alpha = layers[0], layers[1]
	return batch_dot(seq, alpha)

def attentionLayer(windowModel, candidateModel, seqLen, hiddenSize, cl2):
	"""
	Builds an attention model over a context window in light of the candidate embedding
	"""
	def getSeq(layer):
		return layer[:, :seqLen, :]
	
	#build the attention model for the window aka sequence half
	#TODO remove
	print("sequence length {}".format(seqLen))
	print("keras shape {}".format(windowModel._keras_shape))

	windowAttention = TimeDistributed(Dense(hiddenSize, W_regularizer=l2(cl2)))(windowModel)

	#build the attention model for the candidate embedding
	candidateProj = Dense(hiddenSize, W_regularizer=l2(cl2))(candidateModel)
	candidateAttention = RepeatVector(seqLen)(candidateProj)

	#add the two halves together
	attention = merge([windowAttention, candidateAttention], mode="sum")

	#apply an activation function
	nlAttention = Activation("tanh")(attention)

	#flatten the model
	alphaInit = TimeDistributed(Dense(1, activation="linear"))(nlAttention)
	alphaFlat = Flatten()(alphaInit)

	#apply soft max to create the "alpha values"
	alpha = Dense(seqLen, activation="softmax")(alphaFlat)
	alpha = Reshape((seqLen,1))(alpha)

	#TODO remove
	print("alpha shape {}".format(alpha._keras_shape))

	#apply the attention to the orginal sequence, creating a reprentation for the entire
	#sequence, and embed the sequence with attention
	seqTrans = Permute((2, 1))(windowModel)  

	#TODO remove
	print("seqTrans shape {}".format(seqTrans._keras_shape))

	proj = merge([seqTrans, alpha], output_shape=(hiddenSize, 1), mode=seqCombine)

	seqEmbed = Reshape((hiddenSize,))(proj)

	#combine the embeddings and transform them
	finalSeqProj = Dense(hiddenSize, W_regularizer=l2(cl2))(seqEmbed)

	#combine the focused sequences with the candidate vector
	combined = merge([finalSeqProj, candidateProj], mode="sum")
	embedding = Activation("tanh")(combined)

	#return the attention embedding
	return embedding

def buildAttentionModel(numClasses, seqLen, vecDim, contextDim, eventMap):
	"""
	Constructs a model with two attention portions
	"""
	hidden = 512
	rnnHidden = 128
	denseDim = 256
	cl2 = .001
	drop = .5
	convSize = 2
	maxSize = 2
	shape = (seqLen, vecDim)

	#left context model
	leftInput = Input(shape=shape)
	lconv = Conv1D(denseDim, convSize, W_regularizer=l2(cl2))(leftInput)
	lmax = MaxPooling1D(maxSize)(lconv)
	left = GRU(rnnHidden, W_regularizer=l2(cl2), U_regularizer=l2(cl2), dropout_W=drop, dropout_U=drop, return_sequences=True)(lmax)

	#the right context model
	rightInput = Input(shape=shape)
	rconv = Conv1D(denseDim, convSize, W_regularizer=l2(cl2))(rightInput)
	rmax = MaxPooling1D(maxSize)(rconv)
	right = GRU(rnnHidden, W_regularizer=l2(cl2), U_regularizer=l2(cl2), dropout_W=drop, dropout_U=drop, return_sequences=True)(rmax)

	#the model for the candidate
	cInput = Input(shape=(contextDim,))
	cdense = Dense(denseDim)(cInput)
	crelu = LeakyReLU(.01)(cdense)
	context = Dropout(drop)(crelu)

	#get the length after max pooling
	shortLen = right._keras_shape[1]

	#left attention 
	leftAttn = attentionLayer(left, context, shortLen, rnnHidden, cl2)

	#right attention 
	rightAttn = attentionLayer(right, context, shortLen, rnnHidden, cl2)

	joint = merge([leftAttn, rightAttn, context], mode="concat")
	#model = Sequential()
	#model.add(Merge([leftAttn, rightAttn, context], mode="concat"))

	jointDense = Dense(denseDim, W_regularizer=l2(cl2))(joint)
	#model.add(Dense(denseDim, W_regularizer=l2(cl2)))

	jointRelu = LeakyReLU(.01)(jointDense)
	#model.add(LeakyReLU(.01))
	#model.add(MaxoutDense(denseDim, W_regularizer=l2(cl2)))

	jointDrop = Dropout(drop)(jointRelu)
	#model.add(Dropout(drop))

	jointClass = Dense(numClasses)(jointDrop)
	#model.add(Dense(numClasses))

	jointSoft = Activation("softmax")(jointClass)
	#model.add(Activation("softmax"))

	model = Model(input=[leftInput, rightInput, cInput], output=jointSoft)

	#modelWrapper = Sequential()
	#modelWrapper.add(model)

	#NOTE hack to fix the order of the inputs
	#[left, context, right] = modelWrapper.inputs
	#modelWrapper.inputs = [left,right,context]

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', microF1(eventMap)])

	return model

def buildBaseCNN(seqLen, initWeights, contextDim):
	"""
	The base CNN, returns partial model
	"""
	cnnHidden = 150
	rnnHidden = 16
	cl2 = .000
	convSize = [2,3,4,5]

	entDim = 20
	distDim = 2
	
	#NOTE hard coded for dataset
	fullEmbeddingDim = 300 + entDim + distDim 

	shape = (seqLen, fullEmbeddingDim)

	model= Sequential()

	if type(initWeights) != dict:
		init = {rm.wordEmbName:[initWeights]}
	else:
		init = initWeights

	#NOTE hardcoded for dataset
	emb = tripleEmbedding(init, seqLen, entDim, distDim, 12, 13)
	cnn = multiCNN(shape, cnnHidden, convSize, cl2, 0, init)

	model.add(emb)
	model.add(cnn)

	return model

def buildCNNEmbModel(numClasses, seqLen, contextDim, initWeights, eventMap):
	"""
	Reproduces an CNN with embedding
	"""
	drop = .5
	model = buildBaseCNN(seqLen, initWeights, 0)
	model.add(Dropout(drop))
	
	model.add(Dense(numClasses, W_constraint=MaxNorm(3.0)))
	model.add(Activation("softmax"))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy', microF1(eventMap)])

	model.summary()

	return model

def buildMultiEmbModel(numClasses, seqLen, contextDim, initWeights, eventMap):
	"""
	Does a RNN over a CNN model with context information
	"""
	drop = .5
	model = buildBaseCNN(seqLen, initWeights, 0)

	simple = Input(shape=(contextDim,))
	level2 = Model(input=simple, output=simple)

	level3 = Sequential()

	level3.add(Merge([model, level2], mode="concat"))
	level3.add(Dropout(drop))
	
	level3.add(Dense(numClasses, W_constraint=MaxNorm(3.0)))
	level3.add(Activation("softmax"))

	level3.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy', microF1(eventMap)])

	level3.summary()

	return level3

def buildCNNModel(numClasses, seqLen, vecDim, contextDim, eventMap):
	"""
	Reproduces an CNN
	"""
	hidden = 512
	cnnHidden = 150
	denseDim = 256
	cl2 = .0
	drop = .5
	convSize = [2,3,4,5]
	shape = (seqLen, vecDim)

	model= Sequential()

	cnn = multiCNN(shape, cnnHidden, convSize, cl2)
	model.add(cnn)

	#TODO remove
	print("cnn shape {}".format(cnn.output_shape))

	#model.add(Flatten())
	#model.add(BatchNormalization())
	model.add(Dropout(drop))
	model.add(Dense(numClasses))
	model.add(Activation("softmax"))

	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy', microF1(eventMap)])

	print(model.summary())

	return model

def buildModel(numClasses, seqLen, vecDim, contextDim, eventMap):
	"""
	Builds a GRU model on top of word embeddings, and doc2vec
	"""
	hidden = 512
	rnnHidden = 128
	denseDim = 256
	cl2 = .001
	drop = .5
	convSize = 2
	shape = (seqLen, vecDim)

	window = Sequential()

	window.add(Conv1D(denseDim, convSize, W_regularizer=l2(cl2), input_shape=shape))
	window.add(MaxPooling1D(convSize))

	#left.add(Conv1D(denseDim, convSize, W_regularizer=l2(cl2)))
	#left.add(MaxPooling1D(convSize))

	window.add(Bidirectional(GRU(rnnHidden, W_regularizer=l2(cl2), U_regularizer=l2(cl2), dropout_W=drop, dropout_U=drop)))
		
	context = Sequential()
	context.add(Dense(denseDim, input_shape=(contextDim,)))
	context.add(LeakyReLU(.01))
	context.add(Dropout(drop))
	#context.add(Reshape((contextDim,), input_shape=(contextDim,)))

	model = Sequential()
	model.add(Merge([window, context], mode="concat"))

	model.add(Dense(denseDim, W_regularizer=l2(cl2)))
	model.add(LeakyReLU(.01))
	model.add(Dropout(drop))

	model.add(Dense(numClasses))
	model.add(Activation("softmax"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', microF1(eventMap)])

	return model

def buildSplitModel(numClasses, seqLen, vecDim, contextDim, eventMap):
	"""
	Builds a GRU model on top of word embeddings, and doc2vec
	"""
	hidden = 512
	rnnHidden = 128
	denseDim = 256
	cl2 = .001
	drop = .5
	convSize = 2
	maxSize = 2
	shape = (seqLen, vecDim)

	left = Sequential()
	#left.add(GRU(rnnHidden, W_regularizer=l2(cl2), U_regularizer=l2(cl2), dropout_W=drop, dropout_U=drop, input_shape=shape, return_sequences=True))

	left.add(Conv1D(denseDim, convSize, W_regularizer=l2(cl2), input_shape=shape))
	left.add(MaxPooling1D(maxSize))

	#left.add(Conv1D(denseDim, convSize, W_regularizer=l2(cl2)))
	#left.add(MaxPooling1D(convSize))

	left.add(GRU(rnnHidden, W_regularizer=l2(cl2), U_regularizer=l2(cl2), dropout_W=drop, dropout_U=drop))
	
	right = Sequential()
	#right.add(GRU(rnnHidden, W_regularizer=l2(cl2), U_regularizer=l2(cl2), dropout_W=drop, dropout_U=drop, input_shape=shape, return_sequences=True))

	right.add(Conv1D(denseDim, convSize, W_regularizer=l2(cl2), input_shape=shape))
	right.add(MaxPooling1D(maxSize))

	#right.add(Conv1D(denseDim, convSize, W_regularizer=l2(cl2)))
	#right.add(MaxPooling1D(convSize))

	right.add(GRU(rnnHidden, W_regularizer=l2(cl2), U_regularizer=l2(cl2), dropout_W=drop, dropout_U=drop))

	context = Sequential()
	context.add(Dense(denseDim, input_shape=(contextDim,)))
	context.add(LeakyReLU(.01))
	context.add(Dropout(drop))
	
	#do nothing
	#context.add(Reshape((contextDim,), input_shape=(contextDim,)))

	model = Sequential()
	model.add(Merge([left, right, context], mode="concat"))

	model.add(Dense(denseDim, W_regularizer=l2(cl2)))
	model.add(LeakyReLU(.01))
	#model.add(MaxoutDense(denseDim, W_regularizer=l2(cl2)))
	model.add(Dropout(drop))

	model.add(Dense(numClasses))
	model.add(Activation("softmax"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', microF1(eventMap)])

	return model

def buildCNNEmbOutput(numClasses, seqLen, contextDim, eventMap, fullModel, realis=False):
	"""
	Builds a reproduction of the model without the classification layer
	"""
	print("realis {}".format(realis))

	init = {}
	convSize = [2,3,4,5]

	names = [rm.wordEmbName, rm.distEmbName, rm.entEmbName] + [rm.cnnName.format(c) for c in convSize]
	useTrans = False

	#check if the model is using names otherwise look for the layers
	if fullModel.get_layer(rm.wordEmbName) is None:
		fullModel = fullModel.layers[0].layers[0]
		useTrans = True
		names = rm.translation.keys()
	
	#get the weights for each layer
	for name in names:

		layer = fullModel.get_layer(name)

		if layer is None:
			cnnName = rm.cnnModelNameAlt if useTrans else rm.cnnModelName

			layer = fullModel.get_layer(cnnName).get_layer(name)

		print("Layer {} {}".format(name, layer))

		wName = rm.translation[name] if useTrans else name

		init[wName] = layer.get_weights()

	base = buildBaseCNN(seqLen, init, contextDim)

	#add in realis support
	if realis:

		model = Sequential()

		realisIn = Input(shape=(contextDim,))
		realisLayer = Model(input=realisIn, output=realisIn)

		model.add(Merge([base, realisLayer], mode="concat"))

	else:
		model = base

	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	model.summary()

	return model

def loadRealisData(path):
	"""
	Loads saved realis data in the given directory
	"""
	results = []

	#load each named file
	for name in ["training", "dev", "test"]:
		results.append(n.load(join(path, "{}_pred.npy".format(name))))

	return results

def main(args):
	"""
	Runs and evaluates the model
	"""
	#n.random.seed(13)
	n.random.seed(16)
	#n.random.seed(17) #better for realis14
	#n.random.seed(20)

	print("Reading the data")
	dataDict = loadData(args.f, args.s)

	useBothHalves = args.full
	useEmb = args.emb or args.full

	if args.o:
		mkdir(args.o)

	#unpack the data
	if useEmb:
		trainData = setupEmbeddings(dataDict["train_x"], useBothHalves)
		devData = setupEmbeddings(dataDict["dev_x"], useBothHalves)
		testData = setupEmbeddings(dataDict["test_x"], useBothHalves)
	else:
		trainData = dataDict["train_x"]
		devData = dataDict["dev_x"]
		testData = dataDict["test_x"]

	rawTrainingLabels = dataDict["train_y"] 
	rawDevLabels = dataDict["dev_y"] 
	rawTestingLabels = dataDict["test_y"] 
	
	#make the event map
	eventMap = load(open(args.m))

	trainingLabels = eventMap.namesToMatrix(rawTrainingLabels)
	devLabels = eventMap.namesToMatrix(rawDevLabels)
	testingLabels = eventMap.namesToMatrix(rawTestingLabels)

	if useEmb:
		(samples, seqLen) = trainData[0].shape
	else:
		(samples, seqLen, dim) = trainData[0].shape

	print(trainData[0].shape)

	if args.s:
		(rightSamples, contextDim) = trainData[2].shape
	else:

		if useBothHalves:
			rightSamples = trainData[0].shape[0]
			(_, contextDim) = trainData[-1].shape
		else:
			(rightSamples, contextDim) = trainData[-1].shape

	print("#instances: {}, seq len: {}".format(samples, seqLen))
	print("right side {} {}".format(rightSamples, contextDim))
	print("labels shape {}".format(trainingLabels.shape))

	print("Building the model")
	
	#get the model
	if useEmb:

		w2vPath = "data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"
		indexPath = "data/word_index.p"

		#load the realis data
		if args.realis:
			realisData = loadRealisData(args.realis)
			trainData += [realisData[0]]
			devData += [realisData[1]]
			testData += [realisData[2]]

			(_, contextDim) = realisData[0].shape

		#load the weights
		w2v = loadW2V(w2vPath)

		#load the index
		wordIndex = load(open(indexPath))

		#make the initial weights
		initWeights = makeEmbeddingWeights(w2v, wordIndex)

		if args.full or args.realis:
			model = buildMultiEmbModel(len(eventMap), seqLen, contextDim, initWeights, eventMap)

		else:
			model = buildCNNEmbModel(len(eventMap), seqLen, contextDim, initWeights, eventMap)

	else:
		model = buildCNNModel(len(eventMap), seqLen, dim, contextDim, eventMap)

	#train the model
	print("Training the model")
	
	#hard coding class weights...
	weights = defaultdict(lambda: 5.5)
	#weights = defaultdict(lambda: 7.0)
	weights[eventMap.nilIndex()] = 1.0

	#make the logger
	logger = makeLogger(args.o, eventMap)

	model.fit(trainData, trainingLabels, nb_epoch=args.e, batch_size=args.b, 
		validation_data=(devData, devLabels), callbacks=[logger], 
		class_weight=weights)

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
	evaluatePredictions(trainPred, rawTrainingLabels, eventMap)

	print("\n-----Dev Scores------")
	evaluatePredictions(devPred, rawDevLabels, eventMap)

	if args.t:
		testPred = predictClasses(best, testData, args.b)
		print("\n\n-----Test Scores------")
		evaluatePredictions(testPred, rawTestingLabels, eventMap)
	

if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("-e", default=10, type=int, help="Number of epochs")
	parser.add_argument("-b", default=1024, type=int, help="Batch size")
	parser.add_argument("-f", required=True, help="The pickle file with in the input data")
	parser.add_argument("-m", default="data/event_map.p", help="The pickle file with the event map")
	parser.add_argument("-t", action="store_true", help="Evaluate on the testing set")
	parser.add_argument("-s", action="store_true", help="Split the windowed data")
	parser.add_argument("-o", default="", help="The output directory for logging models and error rates")
	parser.add_argument("-emb", action="store_true", help="Use an embedding based model")
	parser.add_argument("-full", action="store_true", help="Use the full model")
	parser.add_argument("-realis", default="", help="Loads the saved realis prediction")

	main(parser.parse_args())
