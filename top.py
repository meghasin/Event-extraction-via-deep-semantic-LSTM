#!/usr/bin/python

"""
Returns the top models
"""

from os.path import join
import re
from argparse import ArgumentParser

from ensemble import parseScores, topK, parseF1Scores
from ml.util import first


def main(args):
	"""
	Compares the top models for both accuracy and F1
	"""
	#f1 = parseF1Scores(args.d)
	acc = parseScores(args.d)

	#topF1 = topK(args.n, f1)
	topAcc = topK(args.n, acc)

	print("---Top F1---")
	#print("\n".join(map(str, topF1)))

	print("---Top Acc---")
	print("\n".join(map(str, topAcc)))

	print("---Unique F1---")
	#print("\n".join(map(str,set(first(topF1)) - set(first(topAcc)))))

if __name__ == "__main__":
	
	parser = ArgumentParser()

	parser.add_argument("-d", required=True, help="A directory with models")
	parser.add_argument("-n", type=int, default=10, help="The number of models to combine")

	main(parser.parse_args())
