#!/usr/bin/env python

from __future__ import division

from os.path import join
from argparse import ArgumentParser
from pickle import load
from collections import defaultdict
import os

import numpy.random as nr
import numpy as n
import keras.backend as b
from sklearn.model_selection import KFold, StratifiedKFold

from basic import makeLogger, loadData, microF1, predictClasses
from main import evaluatePredictions
from windowsMain import buildModel, buildSplitModel, buildCNNEmbModel, buildMultiEmbModel, setupEmbeddings, makeEmbeddingWeights
from ensemble import majorityPredictions
from vectorize import loadW2V
from ml.util import mkdir

class Validator(object):
	"""
	An interface for a cross validator
	"""
	def partition(self, data, labels):
		"""
		Returns a series of partitions for the data and labels
		"""
		raise Exception("Not implmented")

class RandomSplitter(Validator):
	"""
	Randomly splits the data
	"""
	def __init__(self, proportion, splits):
		"""
		Initialize the splitter with a proportion
		"""
		self.proportion = proportion
		self.splits = splits

	def partition(self, data, labels):
		"""
		Returns a series of partitions for the data and labels
		"""
		for i in range(self.splits):
			yield self.makePartition(len(labels))

	def makePartition(self, count):
		"""
		Returns a partition for to use to split the data
		"""
		validPart = (nr.random_sample(count) < self.proportion).astype("int32")
		trainPart = n.ones(count, dtype="int32") - validPart

		return self.toIndexes(trainPart), self.toIndexes(validPart)

	def toIndexes(self, binaryVector):
		"""
		Turns a binary vector into a list of indexes
		"""
		return n.array([i for i,v in enumerate(binaryVector) if v], dtype="int32")

class StandardSplitter(Validator):
	"""
	Uses deterministic splitting
	"""
	def __init__(self, splits):
		"""
		Initialize the splitter with a number of splits to perform
		"""
		self.kfold = KFold(splits)

	def partition(self, data, labels):
		"""
		Returns a series of partitions for the data and labels
		"""
		return self.kfold.split(labels)

class StratifiedSplitter(Validator):
	"""
	Splits the data and keeps an even proportion of label type per split
	"""
	def __init__(self, splits):
		"""
		Initialize the splitter with a number of splits to perform
		"""
		self.kfold = StratifiedKFold(splits)

	def partition(self, data, labels):
		"""
		Returns a series of partitions for the data and labels
		"""
		#TODO remove
		print("label shape {}".format(labels.shape))
		return self.kfold.split(data[0], labels)

def partitionData(data, labels, partition):
	"""
	Splits each element of the data according to the partion
	"""
	return [s[partition] for s in data], labels[partition]

def trainModel(data, labels, vData, vLabels, logger, params):
	"""
	Fits a model to the given data with the given
	"""
	#save the random seed
	logger.logSeed()

	#create the model
	if params.emb:
		
		if params.useBothHalves:
			model = buildMultiEmbModel(params.numClasses(), params.windowSize, params.contextSize, params.wordWeights, params.eventMap)
		else:
			model = buildCNNEmbModel(params.numClasses(), params.windowSize, params.contextSize, params.wordWeights, params.eventMap)

	elif params.split:
		model = buildSplitModel(params.numClasses(), params.windowSize, params.wordSize, params.contextSize, params.eventMap)
		#model = buildAttentionModel(params.numClasses(), params.windowSize, params.wordSize, params.contextSize, params.eventMap)
	else:
		model = buildModel(params.numClasses(), params.windowSize, params.wordSize, params.contextSize, params.eventMap)

	#fit the model
	model.fit(data, labels, nb_epoch=params.epochs, batch_size=params.batchSize, 
		validation_data=(vData, vLabels), 
		callbacks=[logger], class_weight=params.weights)

	best = logger.best()

	print("Best Model round: {} val: {}".format(logger.bestModel, logger.bestScore))

	#return the best
	return best, logger.bestModel

def trainOnFold(data, labels, outDir, numModels, partition, params):
	"""
	Trains several models the given data and parition and returns the best one
	"""
	trainPart, devPart = partition

	#partition the data and labels
	trainX, trainY = partitionData(data, labels, trainPart)
	devX, devY = partitionData(data, labels, devPart)

	models = []

	#train multiple models
	for i in range(numModels):

		modelDir = join(outDir, str(i))

		mkdir(modelDir)

		#setup logger
		logger = makeLogger(modelDir, params.eventMap)

		#train model
		model, index = trainModel(trainX, trainY, devX, devY, logger, params)

		#make predictions
		pred = predictClasses(model, devX, params.batchSize)

		#evaluate using F1
		score = evaluatePredictions(pred, params.eventMap.matrixToNames(devY), params.eventMap, False)

		models.append((score, i, index, model))

		#need to clean up after building a model
		b.clear_session()

	#return best model
	return max(models)

def crossTrain(data, labels, outDir, modelsPer, validator, params):
	"""
	Does cross validated based training
	"""
	completed = 0
	results = []

	out = open(join(outDir, "log.txt"), "a")

	#for each fold split the data and train models keeping the best
	for i, partition in enumerate(validator.partition(data,params.eventMap.matrixToIndex(labels))):

		if params.limit and completed == params.limit:
			return results

		print("Fold {}".format(i))

		#create the output directory
		foldDir = join(outDir, "fold{}".format(i))

		if not os.access(foldDir, os.F_OK):
			mkdir(foldDir)

			print("Training Size {}, Dev Size {}".format(len(partition[0]), len(partition[1])))

			#train models
			bestScore, rnd, epoch, best = trainOnFold(data, labels, foldDir, modelsPer, partition, params)

			print("Best Score {}".format(bestScore))

			out.write("Round {}, Epoch {}, Score {}\n".format(rnd, epoch, bestScore))
			out.flush()
			os.fsync(out)

			results.append(best)
			completed += 1
		else:
			print("Fold {} already exists".format(i))

	out.close()

	return results

def joinDev(training, tLabels, dev, dLabels):
	"""
	Joins the training and development sets together
	"""
	data = [n.concatenate([t,d]) for t,d in zip(training, dev)]

	return data, n.concatenate([tLabels, dLabels])

class Parameters(object):
	"""
	Holds all the parameters for fitting a model
	"""
	def __init__(self, eventMap):
		"""
		Initalize the parameters
		"""
		self.eventMap = eventMap
		self.windowSize = 0
		self.samples = 0
		self.contextSize = 0
		self.epochs = 0
		self.batchSize = 0
		self.wordSize = 0
		self.split = False
		self.limit = 0
		self.useBothHalves = False
		self.emb = False

		#hard coding class weights...
		self.weights = defaultdict(lambda: 5.0)
		self.weights[eventMap.nilIndex()] = 1.0

	def numClasses(self):
		return len(self.eventMap)

def main(args):
	"""
	Runs and evaluates the model
	"""
	print("Reading the data")
	dataDict = loadData(args.f, args.s)

	useBothHalves = args.full
	useEmb = args.emb or args.full

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

	params = Parameters(eventMap)

	trainingLabels = eventMap.namesToMatrix(rawTrainingLabels)
	devLabels = eventMap.namesToMatrix(rawDevLabels)
	#testingLabels = eventMap.namesToMatrix(rawTestingLabels)

	if args.dev:
		data, labels = joinDev(trainData, trainingLabels, devData, devLabels)

	else:
		data = trainData
		labels = trainingLabels

	params.emb = args.emb
	params.useBothHalves = useBothHalves
	params.samples = data[0].shape[0]
	params.windowSize = data[0].shape[1]
	params.batchSize = args.b
	params.epochs = args.e
	params.split = args.s
	params.limit = args.limit

	if useEmb:
		w2vPath = "data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"
		indexPath = "data/word_index.p"
		
		#load the weights
		w2v = loadW2V(w2vPath)
		
		#load the index
		wordIndex = load(open(indexPath))
		
		#make the initial weights
		params.wordWeights = makeEmbeddingWeights(w2v, wordIndex)

	else:
		params.wordSize = data[0].shape[2]

	if args.s:
		params.contextSize = data[2].shape[1]
	else:

		if useBothHalves:
			params.contextSize = data[-1].shape[1]
		else:
			params.contextSize = data[1].shape[1]

	print("Training")

	if args.std:
		print("Standard Cross Validation")
		validator = StandardSplitter(args.c)
	
	elif args.strat:
		print("Stratified Cross Validation")
		validator = StratifiedSplitter(args.c)

	else:
		print("Random Cross Validation")
		validator = RandomSplitter(args.p, args.c)

	mkdir(args.o)

	models = crossTrain(data, labels, args.o, args.k, validator, params)

	print("Make Predictions")
	
	"""
	#make predictions
	trainPred = majorityPredictions(models, trainData, args.b, len(eventMap))
	devPred = majorityPredictions(models, devData, args.b, len(eventMap))

	print("\nEvalutation")

	print("-----Training Scores-----")
	evaluatePredictions(trainPred, rawTrainingLabels, eventMap)

	print("\n-----Dev Scores------")
	evaluatePredictions(devPred, rawDevLabels, eventMap)

	if args.t:
		testPred = majorityPredictions(models, testData, args.b, len(eventMap))
		
		print("\n\n-----Test Scores------")
		evaluatePredictions(testPred, rawTestingLabels, eventMap)
	"""

if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("-e", default=10, type=int, help="Number of epochs")
	parser.add_argument("-b", default=2048, type=int, help="Batch size")
	parser.add_argument("-f", required=True, help="The pickle file with in the input data")
	parser.add_argument("-m", default="data/event_map.p", help="The pickle file with the event map")
	parser.add_argument("-t", action="store_true", help="Evaluate on the testing set")
	parser.add_argument("-s", action="store_true", help="Split the windowed data")
	parser.add_argument("-o", required=True, help="The output directory for logging models and error rates")
	parser.add_argument("-c", default=10, type=int, help="The number of cross validation folds")
	parser.add_argument("-k", default=10, type=int, help="The number of models to fit per fold")
	parser.add_argument("-p", default=.1, type=float, help="The proportion of the data to use for the holdout set for each fold")
	parser.add_argument("-strat", action="store_true", help="Use stratified cross validation")
	parser.add_argument("-std", action="store_true", help="Uses standard cross validation")
	parser.add_argument("-dev", action="store_true", help="Include the dev files when training")
	parser.add_argument("-limit", type=int, default=0, help="The limit of folds to do")
	parser.add_argument("-emb", action="store_true", help="Use an embedding based model")
	parser.add_argument("-full", action="store_true", help="Use the full model")
		
	main(parser.parse_args())
