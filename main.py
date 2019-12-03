#!/usr/bin/env python

"""
The main module for Event Trigger Identification via Deep Learning
"""
from __future__ import division

from os.path import join
from argparse import ArgumentParser
from pickle import load
from collections import defaultdict

from annotation import NIL_LABEL, EventMap
from basic import buildModel, makeLogger, microF1, loadData

import tensorflow as tf

def calcPrecision(correct, predicted):
	return correct / predicted if predicted > 0 else 0.0

def calcRecall(correct, trueCount):
	return correct / trueCount if trueCount > 0 else 0.0

def calcF1(precision, recall):
	total = precision + recall
	return 2 * (precision * recall) / total if total > 0.0 else 0.0

def summaryStats(gold, predLabels, labels, printOut=True):
	"""
	Computes the F1 Score
	"""
	correct = defaultdict(int)
	predicted = defaultdict(int)
	goldNum = defaultdict(int)
	totalCorrect = 0
	totalPred = 0
	totalTrue = 0

	#count up the stats
	for g,p in zip(gold,predLabels):
		
		#if there is a correct prediction update
		if g == p:
			correct[g] += 1

		#count predicted
		predicted[p] += 1

		#count gold
		goldNum[g] += 1

	#print the results
	for l in sorted(labels):
		stats = {}
		stats["prec"] = calcPrecision(correct[l], predicted[l])
		stats["recall"] = calcRecall(correct[l], goldNum[l])
		stats["f1"] = calcF1(stats["prec"], stats["recall"])
		stats["correct"] = correct[l]
		stats["pnum"] = predicted[l]
		stats["gnum"] = goldNum[l]
		totalCorrect += correct[l]
		totalPred += predicted[l]
		totalTrue += goldNum[l]

		if printOut:
			print("{}\t\t\tp:{correct}/{pnum}={prec},\tr:{correct}/{gnum}={recall},\tf1:{f1}".format(l, **stats))

	#do the total summary
	prec = calcPrecision(totalCorrect, totalPred)
	recall = calcRecall(totalCorrect, totalTrue)
	f1 = calcF1(prec, recall)

	stats = {"correct":totalCorrect, "pnum":totalPred, "gnum":totalTrue, "recall":recall, "prec":prec, "f1":f1}

	if printOut:
		print("Overall,\t\t\tp:{correct}/{pnum}={prec},\tr:{correct}/{gnum}={recall},\tf1:{f1}".format(**stats))

	return f1

def markErrors(gold, predLabels, events):
	"""
	Mark the errors and missing events
	"""
	for g,p,e in zip(gold, predLabels, events):
		
		#if there is a mistake, output it
		if g != p:
		
			#false positive case
			if g == NIL_LABEL:
				print("False positive: {}-{} s: {} t: {}".format(p, g, e.sentenceId, e.tokenId))

			#missing prediction
			elif p == NIL_LABEL:
				print("Missing: {}-{} s: {} t: {}".format(p, g, e.sentenceId, e.tokenId))

			#confusion
			else:	
				print("Confusion: {}-{} s: {} t: {}".format(p, g, e.sentenceId, e.tokenId))

def evaluatePredictions(predicted, gold, eventMap, printOut=True):
	"""
	Evaluates and prints the results of the predictions
	"""
	#replace the indexs with the string labels
	predLabels = eventMap.toNames(predicted)

	#skip the Nil label
	labels = eventMap.eventLabels()

	return summaryStats(gold, predLabels, labels, printOut)

def main(args):
	"""
	Runs and evaluates the model
	"""
	#show gpu connection info
	#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	
	print("Reading the data")
	dataDict = loadData(args.f)

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

	(samples, length) = trainingData.shape

	print("#instances: {}, vector length: {}".format(samples, length))
	
	print("Building the model")
	
	#get the model
	model = buildModel(length, len(eventMap), microF1(eventMap))
	#model = buildModel(length, len(eventMap))
	print(model.summary())

	print("Training the model")
	#train the model
	#TODO include F1 metric
	#TODO try 1/cube root p for weights
	#TODO write out parameters to logger

	#hard coding class weights...
	#weights = defaultdict(lambda: 49.0)
	#weights = defaultdict(lambda: 1.0)
	#weights = defaultdict(lambda: 25.0)
	weights = defaultdict(lambda: 10.0)
	weights[eventMap.nilIndex()] = 1.0

	#make the logger
	logger = makeLogger(args.o, eventMap)

	model.fit(trainingData, trainingLabels, nb_epoch=args.e, batch_size=args.b, validation_data=(devData, devLabels), class_weight=weights, callbacks=[logger])

	#get the best model
	best = logger.best()

	print("Best Model round: {} val: {}".format(logger.bestModel, logger.bestScore))

	print("Make Predictions")
	#make predictions
	trainPred = best.predict_classes(trainingData, batch_size=args.b)
	devPred = best.predict_classes(devData, batch_size=args.b)

	print("\nEvalutation")
	#evaluate the model

	print("-----Training Scores-----")
	evaluatePredictions(trainPred, rawTrainingLabels, eventMap)

	print("\n-----Dev Scores------")
	evaluatePredictions(devPred, rawDevLabels, eventMap)

	if args.t:
		testPred = best.predict_classes(testData, batch_size=args.b)
		print("\n\n-----Test Scores------")
		evaluatePredictions(testPred, rawTestingLabels, eventMap)

	print("STD eval {}".format(best.evaluate(devData, devLabels)))
if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("-e", default=10, type=int, help="Number of epochs")
	parser.add_argument("-b", default=1024, type=int, help="Batch size")
	parser.add_argument("-f", required=True, help="The pickle file with in the input data")
	parser.add_argument("-m", default="data/event_map.p", help="The pickle file with the event map")
	parser.add_argument("-t", action="store_true", help="Evaluate on the testing set")
	parser.add_argument("-o", default="", help="The output directory for logging models and error rates")
		
	main(parser.parse_args())
