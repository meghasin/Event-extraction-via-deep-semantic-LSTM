#!/usr/bin/env python

"""
Combines multiple models for better accuracy
"""
from pickle import load
from os.path import join, basename
from os import listdir, walk
from argparse import ArgumentParser
import re
from collections import namedtuple

from keras.layers import Merge
from keras.models import Sequential
import numpy as n

from main import evaluatePredictions
from basic import loadData, loadModel
from windowsMain import setupEmbeddings

LOG = "log.txt"

def parseScores(directory):
	"""
	Read in all the scores and make a map between model and score
	"""
	results = {}
	start = False

	with open(join(directory, "log.txt")) as out:
		for line in out:

			if re.match(r"^[0-9]+:", line):
				start = True

			if start:
				parts = line.split(" ")
				modelId = int(parts[0].strip(":"))
				score = float(parts[-1])

				results[join(directory, "model_{}.h5".format(modelId))] = score

	return results

def parseF1Scores(directory):
	"""
	Read in all the scores and make a map between model and score
	"""
	results = {}

	with open(join(directory, "f1_scores.txt")) as out:
		for line in out:
			parts = line.split(" - ")
			modelId = re.match(r"model_([0-9]+).h5", parts[0]).group(1)
			score = float(parts[-1])

			results[join(directory, "model_{}.h5".format(modelId))] = score

	return results

def loadScores(directories, parser):
	"""
	Loads each model
	"""
	results = {}

	#for each sub folder, look up the score of the model
	for directory in directories:
		for path, dirs, names in walk(directory):
			if LOG in names:
				results.update(parser(path))

	return results

def topK(k, modelScores):
	"""
	Finds the top k models
	"""
	return list(reversed(sorted(modelScores.items(), key=lambda m: m[1])))[:k]

def loadModels(modelList, eventMap):
	"""
	Loads the top models
	"""
	models = []

	for model, score in modelList:
		print("top {}, score {}".format(model, score))
		models.append(loadModel(model, eventMap))
		
	return models

def majorityPredictions(models, data, batchSize, numClasses):
	"""
	Count up the prediction for each instance and make the majority decision
	"""
	if type(data) == list:
		size = len(data[0])
	else:
		size = len(data)

	table = n.zeros( (size, numClasses), dtype="Int32" )
	results = []

	#count up the predictions of the models
	for model in models:
		for i, classIndex in enumerate(model.predict_classes(data, batch_size=batchSize)):
			table[(i,classIndex)] += 1

	#pick the majority decision
	for vec in table:
		index, count = max(enumerate(vec), key=lambda x:x[1])

		results.append(index)
	
	return n.array(results)

def loadBest(directory, eventMap, k=1, parser=parseScores):
	"""
	Returns the k best models
	"""
	return loadModels(topK(k, loadScores(directory, parser)), eventMap)

LoggedModel = namedtuple("LoggedModel", ["fold", "round", "epoch", "score"])

def readLogFile(path):
	"""
	Returns a list of the best model info
	"""
	results = []
	
	#read in the best model for each fold
	for i,line in enumerate(open(path)):
		rnd, epoch, score = [c.split(" ")[1] for c in line.strip().split(", ")]

		results.append(LoggedModel(i, rnd, epoch, score))

	return results

def loadEnsemble(directory, eventMap):
	"""
	Loads a predefined ensemble from the file system
	"""
	models = []

	#for each model in the log file, load it
	for entry in readLogFile(join(directory, "log.txt")):
		models.append(loadModel(join(directory, "fold{}/{}/model_{}.h5".format(entry.fold, entry.round, entry.epoch)), eventMap))

	return models

def main(args):
	"""
	Builds and evaluates an ensemble
	"""
	useBothHalves = args.full

	#read the data
	print("Reading the data")
	dataDict = loadData(args.f, args.s)

	trainingData = dataDict["train_x"]
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

	if args.emb:
		trainingData = setupEmbeddings(trainingData, useBothHalves)
		devData = setupEmbeddings(devData, useBothHalves)
		testData = setupEmbeddings(testData, useBothHalves)

	if args.f1:
		parser = parseF1Scores
	else:
		parser = parseScores

	#load the pre-made ensemble
	if args.e:
		models = loadEnsemble(args.e, eventMap)

	#find the top models
	else:
		models = loadBest(args.d, eventMap, args.n, parser)

	#evalutate the ensemble
	#make predictions
	trainPred = majorityPredictions(models, trainingData, args.b, len(eventMap))
	devPred = majorityPredictions(models, devData, args.b, len(eventMap))

	print("-----Training Scores-----")
	evaluatePredictions(trainPred, rawTrainingLabels, eventMap)

	print("\n-----Dev Scores------")
	evaluatePredictions(devPred, rawDevLabels, eventMap)

	if args.t:
		testPred = majorityPredictions(models, testData, args.b, len(eventMap))
		print("\n\n-----Test Scores------")
		evaluatePredictions(testPred, rawTestingLabels, eventMap)

if __name__ == "__main__":
	
	parser = ArgumentParser()

	parser.add_argument("-f", required=True, help="The data file to use for evaluation")
	parser.add_argument("-b", default=1024, type=int, help="Batch size")
	parser.add_argument("-d", default=[], action="append", help="A directory with models")
	parser.add_argument("-n", type=int, default=10, help="The number of models to combine")
	parser.add_argument("-m", default="data/event_map.p", help="The pickle file with the event map")
	parser.add_argument("-f1", action="store_true", help="Use F1 to select the top models")
	parser.add_argument("-s", action="store_true", help="Split the windowed data")
	parser.add_argument("-t", action="store_true", help="Evaluate on the testing set")
	parser.add_argument("-e", default="", help="A directory containing a premade ensemble")
	parser.add_argument("-emb", action="store_true", help="Prepare the embedding data")
	parser.add_argument("-full", action="store_true", help="Use the full model")

	main(parser.parse_args())
