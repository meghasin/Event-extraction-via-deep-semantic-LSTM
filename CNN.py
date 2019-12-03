'''
Created on Mar 7, 2017

@author: purbasha
'''
from __future__ import print_function
from tensorflow.python import *
import tensorflow as tf
import numpy as np
import sys
from pickle import load
from theano.gradient import tensor
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import *
from keras.layers import Dense, Dropout, Activation, Input, merge, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.datasets import imdb
from os.path import join
from seqMain import *
from keras.layers.wrappers import TimeDistributed


batch_size = 64
hidden_dims = 250
nb_epoch = 20

def word_cnn_buildmodel(numclasses, seqLen, vecDim, rightDim):
# set parameters:
    hidden = 128
    denseDim = 256
    cl2 = .001
    drop = .2  
    print('Build model...')
    left = Sequential()
    left.add(Conv1D(batch_size,3,input_shape=(seqLen, vecDim)))
    left.add(Activation('relu'))
    left.add(Conv1D(batch_size,3,activation='relu'))
    left.add(MaxPooling1D(3,2,'valid'))
    left.add(Bidirectional(LSTM(hidden, return_sequences=True), input_shape=(seqLen, vecDim)))
    left.add(Flatten())
    right = Sequential()
    right.add(Dense(denseDim, input_shape=(rightDim,)))
    right.add(Activation("relu"))  
    
    model = Sequential()
# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
    model.add(Merge([left, right], mode="concat"))
    

# We add a vanilla hidden layer:
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
    '''
    model.add((Dense(50)))
    
    
    x1 = Input(shape=(seqLen, vecDim), name='x1')
    x2 = Input(shape=(seqLen, vecDim), name='x2')
    y1 = model(x1)
    y2 = model(x2)
    '''
   # merged_vector = merge([x1, x2], mode='concat')
   # model.add(TimeDistributed(Dense(numclasses)))(merged_vector)
    #model.add(TimeDistributed(Activation("softmax")))
    print ('seq:::', seqLen,'\t','dim:::',vecDim)

    model.add(Dense(numclasses))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    print(model.summary())
    return model

def main(argdoc):
    """
    Runs and evaluates the model
    """
    global nb_epoch, batch_size
    print("Reading the data")
    dataDict = loadDir(argdoc)
    trainData = list(dataDict["train_x"])
    devData = list(dataDict["dev_x"])
    testData = list(dataDict["test_x"])

    rawTrainingLabels = dataDict["train_y"] 
    rawDevLabels = dataDict["dev_y"] 
    rawTestingLabels = dataDict["test_y"] 
    
    #wordIndex = dataDict["word_index"]
    
    #make the event map
    eventMap = load(open('data/event_map.p'))

    trainingLabels = eventMap.namesToMatrix(rawTrainingLabels)
    devLabels = eventMap.namesToMatrix(rawDevLabels)
    testingLabels = eventMap.namesToMatrix(rawTestingLabels)

    
    (samples, seqLen, dim) = trainData[0].shape
    (rightSamples, rightDim) = trainData[1].shape

    print("#instances: {}, seq len: {} vector length: {}".format(samples, seqLen, dim))
    print("right side {} {}".format(rightSamples, rightDim))
    #print("#instances: {}".format(len(trainLeftData)))
    
    print("Building the model")
    
    #get the model
    model = word_cnn_buildmodel(len(eventMap), seqLen, dim, rightDim)


    print("Training the model")
    #train the model

    weights = defaultdict(lambda: 10.0)
    weights[eventMap.nilIndex()] = 1.0

    #make the logger
    logger = makeLogger('data')

    #TODO remove
    """
    trainLeftData = n.array( [ [1,2,3], [1,5] ] )
    trainingLabels = n.array(seqNamesToMatrix( [ ["NIL", "NIL", "Attack"], ["NIL", "Attack"] ], eventMap ))
    devLeftData = trainLeftData
    devLabels = trainingLabels
    """
    
    #model.fit(n.asarray(trainLeftData), n.asarray(trainingLabels), nb_epoch=args.e, batch_size=args.b, validation_data=(devLeftData, devLabels), class_weight=weights, callbacks=[logger])
    model.fit(trainData, trainingLabels, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(devData, devLabels), callbacks=[logger])

    #get the best model
    best = logger.best()

    print("Best Model round: {} val: {}".format(logger.bestModel, logger.bestScore))

    trainPred = best.predict_classes(trainData, batch_size=args.b)
    devPred = best.predict_classes(devData, batch_size=args.b)

    print("\nEvalutation")
    #evaluate the model

    print("-----Training Scores-----")
    evaluatePredictions(trainPred, rawTrainingLabels, eventMap)

    print("\n-----Dev Scores------")
    evaluatePredictions(devPred, rawDevLabels, eventMap)

    testPred = best.predict_classes(testData, batch_size=args.b)
    print("\n\n-----Test Scores------")
    evaluatePredictions(testPred, rawTestingLabels, eventMap)

if __name__ == "__main__":

    main(sys.argv[1])
