from __future__ import absolute_import
import csv
import json
import numpy
import pickle
from time import time
from scipy.sparse import load_npz
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from log import Log

# This attempts to perform LDA on the text of the notes from ArchivesSpace.

def loadNotes(filename):
	# This returns a list of pixel values, split by the number of pixels in
	data = []
	with open(filename) as file:
		for line in file:
			data.append(line)
	file.closed
	return data

if __name__ == "__main__":

	log = Log('logs/load-and-recommend.log')
	log.message('Loading saved data')

	# TF
	log.message("1a. TF")
	tf = load_npz('data/output/tf.npz')
	log.summarizeTF(tf)

	# TF Vectorizer
	log.message('1b. TF Vectorizer')
	tf_vectorizer = pickle.load(open('data/output/tf_vectorizer.pk', 'rb'))
	log.message(str(type(tf_vectorizer)))
	log.message('')

	# Model
	log.message("2. Model")
	lda = pickle.load(open('data/output/lda_model.pk', 'rb'))
	log.summarizeModel(lda)

	# Universe
	log.message("3. Universe")
	universe = pickle.load(open('data/output/universe.pk', 'rb'))
	universe_expanded = pickle.load(open('data/output/universe_expanded.pk', 'rb'))
	# universe = numpy.fromfile('data/output/trained-from-notes.dat')
	log.summarizeUniverse(universe)

	# Now trying to submit a new document to receive recommendations
	log.message('=================================================================')
	log.message('=================================================================')
	log.message('=================================================================')
	log.message('How about arbitrary input?')
	trial = ['Distinctive Collections collects, preserves, and fosters the use of unique and rare materials such as tangible and digital archives, manuscripts, ephemera, artists’ books, and more. With these collections the Libraries seeks to cultivate an interest in the past, present, and future; the humanistic and the scientific; and the physical and the digital in order to inspire and enable research, learning, experimentation, and play for a diverse community of users.']
	# trial = ['Noam Chomsky is a linguist and political activist. The Noam Chomsky personal archives consists of Chomsky s lecture, speaking, and travel notes, writings, correspondence, research materials, collected serial publications, and other materials documenting his life and work. The collection is divided into six series: Biographical Materials, Correspondence, Writings and Films, Background and Clippings, Speaking Engagements and Travel']
	# trial = ['Collected papers of linguistics and humanities professor Noam Chomsky']
	log.message(str(trial))
	log.message('\n\n')
	trial_matrix = tf_vectorizer.transform(trial)
	log.message(str(type(trial_matrix)))
	log.message(str(trial_matrix))
	trial_affinity = lda.transform(trial_matrix)
	log.message(str(trial_affinity[0]))

	# comparing our results to the universe of known data
	log.message('=================================================================')
	log.message('=================================================================')
	log.message('=================================================================')
	log.message('Comparing results to known data')
	dummy = [trial_affinity[0]]
	log.message('Location of search text: ' + str(dummy))
	log.message('')

	# Calculate distance to all documents using cosine_distance
	trial_pairwise = pairwise.cosine_distances(universe, dummy)

	# Join this calculated distance to the expanded universe of documents and
	# their locations. This will give us something we can sort and have
	# meaningful output - because if we just sort the pairwise results, we
	# only have a number.
	universe_pairwise = numpy.append(universe_expanded, trial_pairwise, axis=1)
	log.message('Random record:')
	log.message(str(universe_pairwise[0, 0:2]))
	log.message('Location:        ' + str(universe_pairwise[0, 2:6]))
	log.message('Relevance score: ' + str(universe_pairwise[0, 6]))

	log.message('')

	# Sort the universe_pairwise array by the final column (number 7 right now)
	# comparing our results to the universe of known data
	log.message('=================================================================')
	log.message('=================================================================')
	log.message('=================================================================')
	log.message('You searched for:')
	log.message(str(trial))
	log.message('\n')
	log.message('Recommender results')
	universe_results = universe_pairwise[numpy.argsort(universe_pairwise[:,6])]
	for item in universe_results[:5]:
		log.message(str(item[6]) + ': ' + str(item[0]))
	log.message('')

	log.message('Finished!')
	log.end()
