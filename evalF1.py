#!/usr/bin/python

"""
Evaluate all the models in a directory based on F1
"""
from pickle import load
from os.path import join, dirname, basename
from os import listdir, walk
from argparse import ArgumentParser

import keras.backend as b

from main import evaluatePredictions
from basic import loadData, loadModel
from windowsMain import setupEmbeddings

def modelNames(modelDir):
	"""
	Returns the path name of all the models in the directory
	"""
	for fileName in listdir(modelDir):
		print(fileName)
		if fileName.endswith(".h5"):
			yield join(modelDir, fileName)

def evalModel(evalData, evalLabels, fileName, eventMap, out, batchSize):

	model = loadModel(fileName, eventMap)

	pred = model.predict_classes(evalData, batch_size=batchSize)

	score = evaluatePredictions(pred, evalLabels, eventMap, False)

	out.write("{} - {}\n".format(basename(fileName), score))

	#clear the tensorflow compute graph
	b.clear_session()

def main(args):
	"""
	Builds and evaluates an ensemble
	"""
	LOG = "log.txt"
	F1 = "f1_{}_scores.txt"
	count = 0

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

	if args.t:
		evalData = testData
		evalLabels = rawTestingLabels
		evalFile = F1.format("test")
	else:
		evalData = devData
		evalLabels = rawDevLabels
		evalFile = F1.format("dev")

	if args.emb:
		evalData = setupEmbeddings(evalData)

	#walk the directory for sub-directories without a scores file
	for path, dirs, names in walk(args.d):
	
		print("In dir {}".format(path))

		#if there is a log file and no f1 scores, eval all the models
		if LOG in names and evalFile not in names:
			
			count += 1
			print("Found Models in {}".format(path))

			#open the output file	
			with open(join(path, evalFile), "a") as out:
				for name in sorted(names):
					if name.endswith(".h5"):
						print("\nEvaluating {}\n".format(name))
						evalModel(evalData, evalLabels, join(path,name), eventMap, out, args.b)

		if count == args.l:
			break

if __name__ == "__main__":
	
	parser = ArgumentParser()

	parser.add_argument("-f", required=True, help="The data file to use for evaluation")
	parser.add_argument("-b", default=1024, type=int, help="Batch size")
	parser.add_argument("-d", required=True, help="A directory with models")
	parser.add_argument("-m", default="data/event_map.p", help="The pickle file with the event map")
	parser.add_argument("-t", action="store_true", help="Evaluate on the testing set")
	parser.add_argument("-s", action="store_true", help="Split the data into windows")
	parser.add_argument("-emb", action="store_true", help="Whether or not the data set uses embeddings")
	parser.add_argument("-l", default=5, type=int, help="The limit on the number of directories to evalutate")

	main(parser.parse_args())
