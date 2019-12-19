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
        print(message)
    print()

def summarizeData(data):
	print('data type: ' + str(type(data)))
	print('rows:      ' + str(len(data)))
	print('fields:    ' + str(len(data[0])))
	print('feild name:')
	print(str(data[0]))
	print('sample record:')
	print(str(data[1]))

def summarizeModel(data):
	print('model type:     ' + str(type(data)))
	print('model topics:   ' + str(len(data.components_)))
	print('model features: ' + str(len(data.components_[0])))
	print('sample topic:')
	print(str(data.components_[0]))
	print('model parameters:')
	print(str(data.get_params()))
	print('')

def summarizeTF(data):
	print('matrix type:      ' + str(type(data)))
	print('tf record count:  ' + str(len(data.toarray())))
	print('tf record length: ' + str(len(data.toarray()[0])) + ' features')
	print('sample record:    ' + str(data.toarray()[0]))
	print('matrix excerpt:\n' + str(data.toarray()))
	print('')

def summarizeUniverse(data):
	print('data type: ' + str(type(data)))
	print('data points: ' + str(len(data)))
	print('dimensions:  ' + str(len(data[0])))
	print('sample data: ' + str(data[0]))
	print('excerpt:')
	print(str(data))
	print('')

if __name__ == "__main__":
	n_samples = 800
	n_features = 100
	n_components = 4
	n_top_words = 10

	# Load data
	print("Loading dataset...")
	# data = loadNotes('data/ArchivesSpace-Rect-Objects-Notes.csv')
	# data = loadData('data/aspace_rectangular_objects.csv')
	data = loadNotes('data/aspace_rectangular_title_notes.csv')
	summarizeData(data)
	print('\n')

	# Split into training and test sets
	print("Splitting dataset...")
	data_samples = data[:n_samples]
	print(str(type(data_samples)))
	train, test = train_test_split(
		data,
		train_size=n_samples,
		shuffle=True
	)
	print('Training records: ' + str(len(train)))
	print('Test records:     ' + str(len(test)))
	print('\n')


	# Add MIT to stopwords
	stop_words = text.ENGLISH_STOP_WORDS.union(['mit','http','https'])

	# Vectorization
	print('=================================================================')

	# Generate tf features for use by LDA.
	# max_df removes words that appear in greater than that % of records
	# min_df removes words that appear in only one record
	# max_features determines the number of dimensions in the resulting list
	# stop_words was 'english' but we appended "mit" just above.
	print("Extracting tf features for LDA...")
	train_notes = isolateNotes(train)
	t0 = time()
	tf_vectorizer = CountVectorizer(max_df=0.90, min_df=2,
	                                max_features=n_features,
	                                stop_words=stop_words)
	tf = tf_vectorizer.fit_transform(train_notes)
	print("done in %0.3fs.\n\n" % (time() - t0))

	# Let's inspect the vectorization...
	print('Vectorization output...\n')
	print('tf')
	summarizeTF(tf)
	print('Saving tf...')
	save_npz('data/output/tf.npz', tf)
	print('')

	print('tf_vectorizer')
	print(str(type(tf_vectorizer)))
	print('Saving tf_vectorizer...')
	pickle.dump(tf_vectorizer, open('data/output/tf_vectorizer.pk', 'wb'))
	# save_npz('data/output/tf_vectorizer.npz', tf_vectorizer)

	print('\n\n')


	# Model fitting
	print('=================================================================')

	# Fit the LDA model using tf features.
	print("Fitting LDA models with tf features, "
	      "n_samples=%d and n_features=%d..."
	      % (n_samples, n_features))
	lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
	                                learning_method='online',
	                                learning_offset=50.,
	                                random_state=0)
	t0 = time()
	lda.fit(tf)
	print("done in %0.3fs." % (time() - t0))
	print("Saving model...")
	pickle.dump(lda, open('data/output/lda_model.pk', 'wb'))

	print("\nTopics in LDA model:")
	tf_feature_names = tf_vectorizer.get_feature_names()
	print_top_words(lda, tf_feature_names, n_top_words)


	# Let's inspect the outcome...
	print('Inspecting fitted model...\n')
	# get_params() returns a dictionary - some of these are visible from when
	# LDA was defined above.
	summarizeModel(lda)

	# Applying vectorization to training data
	print('=================================================================')
	print('Classifying training data into universe')

	universe = lda.transform(tf)
	summarizeUniverse(universe)
	pickle.dump(universe, open('data/output/universe.pk', 'wb'))

	# Merging universe of calculated vectores with training data
	print('=================================================================')
	print('Merging universe and training data')

	expanded = numpy.append(train, universe, axis=1)
	summarizeUniverse(expanded)
	pickle.dump(expanded, open('data/output/universe_expanded.pk', 'wb'))

	print('Finished!')
