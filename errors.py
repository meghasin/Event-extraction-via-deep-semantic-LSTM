#!/usr/bin/python

"""
Prints out errors in HTML format
"""
from argparse import ArgumentParser
from os import listdir
from os.path import join
from pickle import load
from collections import defaultdict, Counter
import unicodedata as ud

from ensemble import loadBest, loadEnsemble, majorityPredictions
from ml.dependency import parseId, parseDocument
from basic import loadData, predictClasses
from annotation import NIL_LABEL
from main import evaluatePredictions, markErrors
from windowsMain import setupEmbeddings, loadRealisData
import vectorize as v

MAIN_TEMPLATE = u"""<html>
<head>
	<style>
		.popup {{
			background:#ccc;
			position:relative;
			text-align:center;
			left:0;
			top:10%;
			width:100%;
		}}

	</style>
</head>

<body>
	<p>Key: <span style='color:green'>correct</span>, 
	<span style='color:blue'>missing</span>, 
	<span style='color:red'>wrong</span></p>
	{body}
	<script>
		var tags = document.getElementsByClassName('parent');

		for(i in tags)
		{{
			var par = tags[i];
			
			par.onmouseover = function()
			{{
				var children = this.childNodes

				for(j in children)
				{{
					var child = children[j];
					if(child.className == 'popup')
					{{
						child.style.display = 'block';
					}}
				}}
			}}

			par.onmouseout = function()
			{{
				var children = this.childNodes

				for(j in children)
				{{
					var child = children[j];
					if(child.className == 'popup')
					{{
						child.style.display = 'none';
					}}
				}}
			}}
		}}
	</script>

</body>
</html>
	"""


HEADER_TEMPLATE = "<hr><b>{doc}</b></hr><p>{text}</p>\n"

EVENT_TEMPLATE = """<span class='parent' style='color:{color}'>{word} <span class='popup' style='display: none;'>correct: {correct}, predicted: {predicted}</span> </span>"""
MISSING_WORD_TEMPLATE = """<span style='color:purple'>{word}</span>"""

def safeWord(text):
	try:
		result = ud.normalize("NFKD", text).encode("ascii", "ignore")
	except:
		result = "UNI"

	return result

def renderWord(token, wordModel):
	"""
	Generates the representation for the text
	"""
	safe = safeWord(token.word)

	if not v.getKey(token, wordModel):
		return MISSING_WORD_TEMPLATE.format(word=safe)
	else:
		return safe

def makeAnnoMap(events, predicted):
	"""
	Makes a map of doc -> sentence -> token -> event type
	"""
	anno = defaultdict(lambda: defaultdict( lambda: defaultdict(lambda: None) ) )

	#build the mapping
	for event, predType in zip(events, predicted):
		anno[event.docId][event.sentenceId][event.tokenId] = (event.type, predType)

	return anno

def createHTML(doc, annotations, wordModel):
	"""
	Creates HTML for the given document
	"""
	chunks = []

	docId = doc.id

	#for each token if an event is present or predicted, output if it is correct
	#missing, or wrong
	for token in doc.tokens():

		#get the annotation for the token
		anno = annotations[docId][token.sentenceId][token.id]

		#if there is annotations, generate html for it
		if anno is not None:
			(gold, pred) = anno
			
			color = None
		#if gold != NIL_LABEL:
			#if pred != NIL_LABEL:
				#print("token: {} gold: {} pred: {}".format(token.word, gold, pred))

			#correct
			if gold == pred and gold != NIL_LABEL:
				color = "green"

			#missing
			elif pred == NIL_LABEL and gold != NIL_LABEL:
				color = "blue"
			
			#wrong
			elif gold != pred:
				color = "red"

			if color:
				chunks.append(EVENT_TEMPLATE.format(color=color, correct=gold, predicted=pred, word=safeWord(token.word)))
			else:
				chunks.append(renderWord(token, wordModel))

		#else just output the text
		else:
			chunks.append(renderWord(token, wordModel))

	return HEADER_TEMPLATE.format(doc=docId, text=" ".join(chunks))

def createErrorReports(dataDir, outPath, annotations, wordModel):
	"""
	Creates html error reports
	"""
	#get a set of all the files based on what is annotated
	targetDocs = set(annotations.keys())
	text = ""

	with open(outPath, "w") as out:
		#for each file in the data directory, if it is in the target,
		#create html for it
		for fileName in listdir(dataDir):
			if parseId(fileName) in targetDocs:
				doc = parseDocument(join(dataDir, fileName))

				text += "{}\n".format(createHTML(doc, annotations, wordModel))

		out.write(MAIN_TEMPLATE.format(body=text))

def main(args):
	"""
	Prints out html indicating the errors
	"""
	w2vPath = "data/vectors/word2vec/GoogleNews-vectors-negative300.bin.gz"

	#load the word vector model
	print("Loading vectors")
	wordModel = v.loadW2V(w2vPath)

	print("Reading Data")
	dataDict = loadData(args.f, args.s)

	trainingData = dataDict["train_x"]
	devData = dataDict["dev_x"]
	testData = dataDict["test_x"]

	trainEvents = dataDict["train_events"]
	devEvents = dataDict["dev_events"]
	testEvents = dataDict["test_events"]

	rawTrainingLabels = dataDict["train_y"] 
	rawDevLabels = dataDict["dev_y"] 
	rawTestingLabels = dataDict["test_y"] 
	realisIndex = None

	if args.tr:
		evalData = trainingData 
		evalLabels = rawTrainingLabels
		evalEvents = trainEvents
		ext = "train"
		realisIndex = 0
		
	elif args.t:
		evalData = testData
		evalLabels = rawTestingLabels
		evalEvents = testEvents
		ext = "test"
		realisIndex = 2
	else:
		evalData = devData
		evalLabels = rawDevLabels
		evalEvents = devEvents
		ext = "dev"
		realisIndex = 1

	print("gold events {}".format(len(evalEvents)))

	#load the event map
	eventMap = load(open(args.m))

	print("Loading the model from {}".format(args.a))
	
	if args.emb:
		evalData = setupEmbeddings(evalData, args.full)

		#load the realis data
		if args.realis:
			realisData = loadRealisData(args.realis)
			evalData += [realisData[realisIndex]]

			(_, contextDim) = realisData[0].shape

	#load the ensemble
	if args.e:
		ensemble = loadEnsemble(args.a, eventMap)

		pred = majorityPredictions(ensemble, evalData, args.b, len(eventMap))

	else:
		#load the model
		model = loadBest([args.a], eventMap)[0]

		#make predictions
		pred = predictClasses(model, evalData, args.b)

	print("Creating HTML")

	#make a map predicted vs actual events
	annoMap = makeAnnoMap(evalEvents, eventMap.toNames(pred))

	markErrors(evalLabels, pred, evalEvents)

	createErrorReports(args.d, join(args.a, "{}_errors.html").format(ext), annoMap, wordModel)

if __name__ == "__main__":

	parser = ArgumentParser()
	
	parser.add_argument("-f", required=True, help="The data file to use for evaluation")
	parser.add_argument("-b", default=2048, type=int, help="Batch size")
	parser.add_argument("-d", required=True, help="The directory with annotated files")
	parser.add_argument("-a", required=True, help="The model directory")
	parser.add_argument("-m", default="data/event_map.p", help="The pickle file with the event map")
	parser.add_argument("-t", action="store_true", help="Evaluate on the testing set")
	parser.add_argument("-tr", action="store_true", help="Evaluate on the training set")
	parser.add_argument("-e", action="store_true", help="Use ensemble")
	parser.add_argument("-s", action="store_true", help="Split the windowed data")
	parser.add_argument("-emb", action="store_true", help="Use embedding data")
	parser.add_argument("-full", action="store_true", help="Use the full model")
	parser.add_argument("-realis", default="", help="Loads the saved realis prediction")
	
	main(parser.parse_args())
