from __future__ import absolute_import
import csv
import json
import numpy
import pickle
import sys
from time import time
from scipy.sparse import save_npz
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from log import Log

# This attempts to perform LDA on the text of the notes from ArchivesSpace.

def isolateNotes(data):
	notes = []
	for item in data:
		notes.append(item[1])
	return notes

def loadNotes(filename):
	csv.field_size_limit(512000)
	# This returns a list of pixel values, split by the number of pixels in
	data = []
	with open(filename, newline='') as csvfile:
		file = csv.reader(csvfile)
		for line in file:
			data.append(line)
	return data

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        log.message(message)
    log.message('')

if __name__ == "__main__":
	n_samples = 800
	n_features = 100
	n_components = 4
	n_top_words = 10

	# Initialize log
	log = Log('logs/train-and-save.log')

	# Load data
	log.message('Loading dataset...')
	# data = loadNotes('data/ArchivesSpace-Rect-Objects-Notes.csv')
	# data = loadData('data/aspace_rectangular_objects.csv')
	data = loadNotes('data/aspace_rectangular_title_notes.csv')
	log.summarizeData(data)
	log.message('\n')

	# Split into training and test sets
	log.message('Splitting dataset...')
	data_samples = data[:n_samples]
	log.message(str(type(data_samples)))
	train, test = train_test_split(
		data,
		train_size=n_samples,
		shuffle=True
	)
	log.message('Training records: ' + str(len(train)))
	log.message('Test records:     ' + str(len(test)))
	log.message('\n')


	# Add MIT to stopwords
	stop_words = text.ENGLISH_STOP_WORDS.union(['mit','http','https'])

	# Vectorization
	log.message('=================================================================')

	# Generate tf features for use by LDA.
	# max_df removes words that appear in greater than that % of records
	# min_df removes words that appear in only one record
	# max_features determines the number of dimensions in the resulting list
	# stop_words was 'english' but we appended "mit" just above.
	log.message("Extracting tf features for LDA...")
	train_notes = isolateNotes(train)
	t0 = time()
	tf_vectorizer = CountVectorizer(max_df=0.90, min_df=2,
	                                max_features=n_features,
	                                stop_words=stop_words)
	tf = tf_vectorizer.fit_transform(train_notes)
	log.message("done in %0.3fs.\n\n" % (time() - t0))

	# Let's inspect the vectorization...
	log.message('Vectorization output...\n')
	log.message('tf')
	log.summarizeTF(tf)
	log.message('Saving tf...')
	save_npz('data/output/tf.npz', tf)
	log.message('')

	log.message('tf_vectorizer')
	log.message(str(type(tf_vectorizer)))
	log.message('Saving tf_vectorizer...')
	pickle.dump(tf_vectorizer, open('data/output/tf_vectorizer.pk', 'wb'))
	# save_npz('data/output/tf_vectorizer.npz', tf_vectorizer)

	log.message('\n\n')


	# Model fitting
	log.message('=================================================================')

	# Fit the LDA model using tf features.
	log.message("Fitting LDA models with tf features, "
	      "n_samples=%d and n_features=%d..."
	      % (n_samples, n_features))
	lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
	                                learning_method='online',
	                                learning_offset=50.,
	                                random_state=0)
	t0 = time()
	lda.fit(tf)
	log.message("done in %0.3fs." % (time() - t0))
	log.message("Saving model...")
	pickle.dump(lda, open('data/output/lda_model.pk', 'wb'))

	log.message("\nTopics in LDA model:")
	tf_feature_names = tf_vectorizer.get_feature_names()
	print_top_words(lda, tf_feature_names, n_top_words)


	# Let's inspect the outcome...
	log.message('Inspecting fitted model...\n')
	# get_params() returns a dictionary - some of these are visible from when
	# LDA was defined above.
	log.summarizeModel(lda)

	# Applying vectorization to training data
	log.message('=================================================================')
	log.message('Classifying training data into universe')

	universe = lda.transform(tf)
	log.summarizeUniverse(universe)
	pickle.dump(universe, open('data/output/universe.pk', 'wb'))

	# Merging universe of calculated vectores with training data
	log.message('=================================================================')
	log.message('Merging universe and training data')

	expanded = numpy.append(train, universe, axis=1)
	log.summarizeUniverse(expanded)
	pickle.dump(expanded, open('data/output/universe_expanded.pk', 'wb'))

	log.message('Finished!')
	log.end()
