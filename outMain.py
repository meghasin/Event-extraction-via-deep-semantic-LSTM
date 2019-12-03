#!/usr/bin/env python

"""
Loads a predicts the hidden layer and probability of the given model
"""

from argparse import ArgumentParser
import numpy as n
from pickle import load
from os.path import join
import csv

import vectorize as v
from basic import loadData
from windowsMain import setupEmbeddings, buildCNNEmbOutput, loadRealisData
from ensemble import loadBest
from realisMain import loadEvents

from ml.util import mkdir

def predictEventEmb(model, datasets):
	"""
	Predicts the hidden layer on each dataset
	"""
	return [model.predict(d) for d in datasets]

def writeEventEmb(datasets, names, eventInfo):
	"""
	Writes out all the datasets to the given locations
	"""
	#write each dataset
	for data, name, events in zip(datasets, names, eventInfo):
	
		with open(name, "w") as csvFile:
			
			print("Outshape {}".format(data.shape))
			out = csv.writer(csvFile)

			#write each row out
			for row, event in zip(data, events):
		
				#print("row {}".format(row))
				out.writerow([event.id, event.docId, event.sentenceId, event.tokenId] + [float(i) for i in row])

def main(args):
	"""
	Makes predictions using the loaded model
	"""
	w2vPath = "data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"

	useBothHalves = args.full
	useEmb = args.emb or args.full
	
	print("Reading Data")
	dataDict = loadData(args.f, args.s)

	#unpack the data
	if useEmb:
		trainData = setupEmbeddings(dataDict["train_x"], useBothHalves)
		devData = setupEmbeddings(dataDict["dev_x"], useBothHalves)
		testData = setupEmbeddings(dataDict["test_x"], useBothHalves)
	else:
		trainData = dataDict["train_x"]
		devData = dataDict["dev_x"]
		testData = dataDict["test_x"]
	
	#rawTrainingLabels = dataDict["train_y"] 
	#rawDevLabels = dataDict["dev_y"] 
	#rawTestingLabels = dataDict["test_y"] 

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
	#print("labels shape {}".format(trainingLabels.shape))

	#load the realis data
	if args.realis:
		realisData = loadRealisData(args.realis)
		trainData += [realisData[0]]
		devData += [realisData[1]]
		testData += [realisData[2]]

		(_, contextDim) = realisData[0].shape

	eventMap = load(open(args.m))

	#load the model
	model = loadBest([args.a], eventMap)[0]

	model.summary()

	eventOut = "eventOut"
	eventProbOut = "eventProbOut"
	mkdir(join(args.a, eventOut))
	mkdir(join(args.a, eventProbOut))

	makeNames = lambda p: [join(args.a, p, i) for i in ["training_pred.csv", "dev_pred.csv", "test_pred.csv"]]

	outModel = buildCNNEmbOutput(len(eventMap), seqLen, contextDim, eventMap, model, len(args.realis) > 0)
	
	eventEmb = predictEventEmb(outModel, [trainData, devData, testData])
	eventProbEmb = predictEventEmb(model, [trainData, devData, testData])

	#load event info
	eventInfo = loadEvents(args.f)

	writeEventEmb(eventEmb, makeNames(eventOut), eventInfo)
	writeEventEmb(eventProbEmb, makeNames(eventProbOut), eventInfo)


if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("-b", default=1024, type=int, help="Batch size")
	parser.add_argument("-f", required=True, help="The pickle file with in the input data")
	parser.add_argument("-a", required=True, help="The model directory")
	parser.add_argument("-m", default="data/event_map.p", help="The pickle file with the event map")
	parser.add_argument("-s", action="store_true", help="Split the windowed data")
	parser.add_argument("-emb", action="store_true", help="Use an embedding based model")
	parser.add_argument("-full", action="store_true", help="Use the full model")
	parser.add_argument("-realis", default="", help="Loads the saved realis prediction")

	main(parser.parse_args())
