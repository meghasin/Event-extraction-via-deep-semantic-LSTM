#!/usr/bin/env python

"""
Builds a mapping from event names to indexes
"""

from pickle import load, dump
from argparse import ArgumentParser

from annotation import EventMap

def main(args):
	"""
	Builds the event mapping
	"""
	print("Reading the data")
	dataDict = load(open(args.f))

	rawTrainingLabels = dataDict["train_y"] 
	rawDevLabels = dataDict["dev_y"] 
	rawTestingLabels = dataDict["test_y"] 
	
	#make the event map
	eventMap = EventMap(rawTrainingLabels + rawDevLabels + rawTestingLabels)

	dump(eventMap, open(args.o, "w"))

if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("-f", required=True, help="The pickle file with in the input data")
	parser.add_argument("-o", required=True, help="The file name for the event mapping")
		
	main(parser.parse_args())
