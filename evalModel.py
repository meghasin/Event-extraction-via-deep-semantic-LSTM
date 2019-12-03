#!/usr/bin/python

"""
Loads and evaluates a saved model on either the development or test sets
"""

from argparse import ArgumentParser
from pickle import load

from main import evaluatePredictions
from basic import loadData, predictClasses, loadModel
from windowsMain import setupEmbeddings

def main(args):
	"""
	Loads and evaluates a model
	"""
	print("Reading the data")
	dataDict = loadData(args.f, args.s)

	devData = dataDict["dev_x"]
	testData = dataDict["test_x"]

	if args.emb:
		devData = setupEmbeddings(devData)
		testData = setupEmbeddings(testData)

	devLabels = dataDict["dev_y"] 
	testingLabels = dataDict["test_y"] 
	
	#make the event map
	eventMap = load(open(args.p))

	#load the model
	print("Loading model")
	model = loadModel(args.m, eventMap)

	#pick the correct set to eval on
	if args.t:
		print("Using Testing")
		evalLabels = testingLabels
		evalData = testData

	else:
		print("Using Dev")
		evalLabels = devLabels
		evalData = devData
		
	print("Evaluating")
	#make predictions
	pred = predictClasses(model, evalData, args.b)

	print("\nEvalutation")
	#evaluate the model

	print("-----Scores-----")
	evaluatePredictions(pred, evalLabels, eventMap)


if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("-m", required=True, help="The model file")
	parser.add_argument("-f", required=True, help="The pickle file with in the input data")
	parser.add_argument("-b", default=64, type=int, help="Batch size")
	parser.add_argument("-p", default="data/event_map.p", help="The pickle file with the event map")
	parser.add_argument("-t", action="store_true", help="Evaluate on the testing set")
	parser.add_argument("-s", action="store_true", help="Split the data")
	parser.add_argument("-emb", action="store_true", help="Prepare the embedding data")

	args = parser.parse_args()

	main(args)
