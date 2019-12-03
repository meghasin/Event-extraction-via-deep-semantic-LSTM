#!/usr/bin/env python

"""
Generates document embeddings for Stanford CoreNLP parsed documents
"""

from argparse import ArgumentParser
from os import listdir
from os.path import join

from gensim.models.doc2vec import Doc2Vec

from ml.dependency import parseDocument

def main(inputDir, modelFileName, outFileName):
	"""
	Generates a vector per document based on the given doc2vec model
	"""
	#load the model
	#model = Doc2Vec.load_word2vec_format(modelFileName, binary=True)
	model = Doc2Vec.load(modelFileName)
	startAlpha=0.01
	inferEpoch=1000
	model.workers = 8

	#open the output file
	out = open(outFileName, "w")

	#apply the model to each file
	for name in listdir(inputDir):

		print("Processing {}".format(name))

		#load the document
		doc = parseDocument(join(inputDir,name))

		#get the document text and generate the vector
		vec = model.infer_vector(doc.words(), alpha=startAlpha, steps=inferEpoch)

		#save the results
		out.write("{}: {}\n".format(doc.id, ", ".join(map(str,vec))))
	
	out.close()

if __name__ == "__main__":
	
	parser = ArgumentParser()

	parser.add_argument("-i", required=True, help="The input directory of corenlp docs")
	parser.add_argument("-m", required=True, help="The saved model file")
	parser.add_argument("-o", default="embeddings.txt", help="The output file with text encoded embeddings")

	args = parser.parse_args()

	main(args.i, args.m, args.o)
