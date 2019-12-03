#!/usr/bin/env python

"""
A basic classification model
"""

from __future__ import division

from os.path import join, isdir
from os import remove
from pickle import load
from math import floor, ceil
from collections import Counter

from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from keras.regularizers import l2
from keras.callbacks import Callback
import keras.backend as kb
import numpy as n
import numpy.random as r

VAL_ACC = "val_acc"
VAL_F1 = "val_f1"
TMP_FILE = "tmp_{:.2}.h5"

def microF1(eventMap):
	"""
	Returns a function to evaluate the micro f1 of the 
	"""
	def f1(gold, predicted):
		return kerasF1(gold, predicted, eventMap.nilIndex())
	
	return f1

def kerasF1(rawGold, rawPredicted, nil):
	gold = kb.argmax(rawGold, axis=-1)
	predicted = kb.argmax(rawPredicted, axis=-1)
	events = kb.not_equal(gold, nil)
	predEvents = kb.not_equal(predicted, nil)

	total = sumBool(events)
	predTotal = sumBool(predEvents)
	correct = sumBool(kb.equal(kb.equal(gold, predicted), events))
	recall =  safeMin(maxStat(correct / kb.maximum(total, 1.0)))
	precision = safeMin(maxStat(correct / kb.maximum(predTotal, 1.0)))

	return bottomOut((2 * precision * recall) / (precision + recall))
	#return sumBool(events)

def safeMin(stat):
	return kb.maximum(stat, 0.0000001)

def bottomOut(stat):
	return kb.cast(kb.greater_equal(stat, 0.000001), "float32") * stat

def maxStat(stat):
	return kb.cast(kb.lesser_equal(stat, 1.0), "float32") * stat

def sumBool(vec):
	return kb.sum(kb.cast(vec, "float32"))

def maskIndex(vector, index):
	"""
	Returns the mask for the given index
	"""
	return kb.cast(kb.equal(vector, index), "int32")

def loadData(path, makeSplit=False):
	"""
	Loads the data from either format
	"""
	def l(name):
		return n.load(join(path, name))

	#if the path is a directory, load all the files in it
	if isdir(path):
		results = {}

		results["train_x"] = [l("training_left.p"), l("training_right.p")]
		results["dev_x"] = [l("dev_left.p"), l("dev_right.p")]
		results["test_x"] = [l("test_left.p"), l("test_right.p")]

		results["train_y"] = l("training_labels.p")
		results["dev_y"] = l("dev_labels.p")
		results["test_y"] = l("test_labels.p")

		if makeSplit:
			results["train_x"] = split(results["train_x"])
			results["dev_x"] = split(results["dev_x"])
			results["test_x"] = split(results["test_x"])

		other = load(open(join(path, "info.p")))
		results.update(other)

	#else just load the pickled file
	else:
		with open(path) as dataFile:
			results = load(dataFile)

	return results

def oldweightClasses(labels, nilType, eventWeight):
	"""
	Returns a dictionary with weights per event type that sum to the given
	event weight, the nil type will have a weight of 1
	"""
	#count up all the types
	counts = Counter(labels)

	#sum up the number of examples that aren't nil
	total = sum(v for k,v in counts.items() if k != nilType)

	#compute the weights for the events
	weights = {k:(v*eventWeight/total) for k,v in counts.items() if k != nilType}	

	#set the nil weight
	weights[nilType] = 1.0

	return weights

def weightClasses(labels, nilType, base, alpha):
	"""
	Returns a dictionary with weights per event type 
	base + alpha * cuberoot(n)
	"""
	#count up all the types
	counts = Counter(labels)

	#compute the weights for the events
	weights = {k:(base + (alpha*(v**(-1/3)))) for k,v in counts.items() if k != nilType}	

	#set the nil weight
	weights[nilType] = 1.0

	return weights

def splitData(window):
	"""
	Splits the window into left/right contexts
	"""
	mid = window.shape[1] / 2
	left = int(floor(mid))
	right = int(ceil(mid))

	groups = n.split(window, [left, right], 1)

	return [groups[0], groups[2]]

def split(data):
	"""
	Splits the window portion of the data
	"""
	return splitData(data[0]) + [data[1]]

def buildModel(inputDim, numClasses, metric=None):
	"""
	Builds a model a little more complex than a logistic regression model
	"""
	cl2 = .001
	drop = .2

	metrics = ['accuracy', metric] if metric is not None else ['accuracy']

	model = Sequential()

	model.add(Dense(output_dim=1024, input_shape=(inputDim,), W_regularizer=l2(cl2)))
	model.add(Activation("relu"))
	model.add(Dropout(drop))

	model.add(Dense(output_dim=1024, W_regularizer=l2(cl2)))
	model.add(Activation("relu"))
	model.add(Dropout(drop))

	model.add(Dense(output_dim=numClasses))
	model.add(Activation("softmax"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

	return model

def makeLogger(path, eventMap, criteria=VAL_ACC):
	"""
	Returns a logger or nothing if no path given
	"""
	if path:
		return ModelLogger(path, eventMap, criteria)
	else:
		return ModelLogger(".", eventMap, criteria, True)

def loadModel(fileName, eventMap):
	"""
	Loads the saved model from the given file
	"""
	return load_model(fileName, custom_objects={"f1":microF1(eventMap)})

def predictClasses(model, data, batchSize):
	"""
	Make class predictions with the given model
	"""
	#if the model is a sequential model, just use it's method
	if isinstance(model, Sequential):
		return model.predict_classes(data, batch_size=batchSize)
	else:
		return model.predict(data, batch_size=batchSize).argmax(axis=-1)
		
class ModelLogger(Callback):
	"""
	Logs the models and their performance
	"""
	
	def __init__(self, outDir, eventMap, criteria, keepBest=False):
		"""
		Initialize the logger
		"""
		self.outDir = outDir
		self.logFile = None
		self.round = 1
		self.keepBest = keepBest
		self.bestScore = -1.0
		self.bestModel = -1
		self.eventMap = eventMap
		self.tmpFile = join("tmp",TMP_FILE.format(r.random()))
		self.criteria = criteria

		if not self.keepBest:
			self.logFile = open(join(self.outDir, "log.txt"), "w")

	def on_train_begin(self, logs={}):
		"""
		prepare for logging models
		"""
		#write the seed - maybe not the best?
		self.logFile.write("on train random state: {}".format(r.get_state()))

	def on_epoch_end(self, batch, logs={}):
		"""
		Logs all the training and validation error/accuracy and the models
		to an output directory/log file
		"""
		score = float(logs.get(self.criteria))
		improved = score > self.bestScore

		#check if the new model is an improvement
		if improved:
			#print("old {}, new {}, round {}".format(self.bestScore, score, self.round))

			self.bestScore = score
			self.bestModel = self.round

		#keep only the best model
		if self.keepBest:
			
			#if there is an improvement
			if improved:
				self.model.save(join(self.outDir, self.tmpFile))
		
		#otherwise log everything
		else:
			#save the model
			self.model.save(self.makeFileName(self.round))

			#log all the metrics
			self.logFile.write("{}: loss {loss}, acc {acc}, val_loss {val_loss}, val_acc {val_acc}\n".format(self.round, **logs))

		self.round += 1

	def logSeed(self):
		"""
		Log the random seed used
		"""
		if not self.keepBest:
			self.logFile.write("random state: {}".format(r.get_state()))

	def best(self):
		"""
		Return the best model seen so far
		"""
		if self.keepBest:
			fileName = join(self.outDir, self.tmpFile)
		else:
			fileName = join(self.outDir, self.makeFileName(self.bestModel))

		return loadModel(fileName, self.eventMap)

	def makeFileName(self, roundNum):
		return join(self.outDir, "model_{}.h5".format(roundNum))

	def __del__(self):
		if self.logFile is not None:
			self.logFile.close()

		#if self.keepBest:
			#remove(join(self.outDir, TMP_FILE))

