from __future__ import absolute_import
import csv
from time import time
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# This attempts to perform LDA on the text of the notes from ArchivesSpace.

def loadNotes(filename):
	# This returns a list of pixel values, split by the number of pixels in
	data = []
	with open(filename) as file:
		for line in file:
			data.append(line)
	file.closed
	return data

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

if __name__ == "__main__":
	n_samples = 800
	n_features = 100
	n_components = 4
	n_top_words = 10

	# Load data
	print("Loading dataset...")
	data = loadNotes('data/ArchivesSpace-Rect-Objects-Notes.csv')
	print('\n\n')

	# Split into training and test sets
	print("Splitting dataset...")
	data_samples = data[:n_samples]
	print(str(type(data_samples)))
	train, test = train_test_split(
		data,
		train_size=n_samples,
		shuffle=True
	)
	print(str(type(train)))
	print(str(len(train)))
	print(str(train[0]))
	print(str(type(test)))
	print(str(len(test)))

	# Add MIT to stopwords
	stop_words = text.ENGLISH_STOP_WORDS.union(['mit'])

	# Vectorization
	print('=================================================================')

	# Generate tf features for use by LDA.
	# max_df removes words that appear in greater than that % of records
	# min_df removes words that appear in only one record
	# max_features determines the number of dimensions in the resulting list
	# stop_words was 'english' but we appended "mit" just above.
	print("Extracting tf features for LDA...")
	tf_vectorizer = CountVectorizer(max_df=0.90, min_df=2,
	                                max_features=n_features,
	                                stop_words=stop_words)
	t0 = time()
	tf = tf_vectorizer.fit_transform(train)
	print("done in %0.3fs.\n\n" % (time() - t0))

	# Let's inspect the vectorization...
	print('Vectorization output...\n')
	print(str(type(tf)))
	print(str(tf.toarray()))
	print('tf length:    ' + str(len(tf.toarray())))
	print('tf[0] length: ' + str(len(tf.toarray()[0])))
	print('tf[0]:        ' + str(tf.toarray()[0]))

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

	print("\nTopics in LDA model:")
	tf_feature_names = tf_vectorizer.get_feature_names()
	print_top_words(lda, tf_feature_names, n_top_words)


	# Let's inspect the outcome...
	print('Inspecting fitted model...\n')
	# get_params() returns a dictionary - some of these are visible from when
	# LDA was defined above.
	print(str(type(lda)))
	print(str(lda.get_params()))
	print(str(lda.components_[0]))

	print(str(len(lda.components_)))
	print(str(len(lda.components_[0])))

	# Experimentation
	print('=================================================================')
	print('=================================================================')
	print('=================================================================')
	print('Applying vectorizer to populate universe of records in new space.')

	universe = lda.transform(tf)
	print(str(type(universe)))
	print(str(universe))
	print(str(universe[0]))
	print('----')
	print(str(len(universe)))
	print(str(len(universe[0])))

	# Now trying to submit a new document to receive recommendations
	print('=================================================================')
	print('=================================================================')
	print('=================================================================')
	print('Now we prepare a new record to trigger recommendations')
	trial = [test[0]]
	print(str(trial))

	trial_matrix = tf_vectorizer.transform(trial)
	print(str(type(trial_matrix)))
	print(str(trial_matrix))

	trial_foo = lda.transform(trial_matrix)
	print(str(trial_foo[0]))

	# Now trying to submit a new document to receive recommendations
	print('=================================================================')
	print('=================================================================')
	print('=================================================================')
	print('How about arbitrary input?')
	trial = ['Distinctive Collections collects, preserves, and fosters the use of unique and rare materials such as tangible and digital archives, manuscripts, ephemera, artists’ books, and more. With these collections the Libraries seeks to cultivate an interest in the past, present, and future; the humanistic and the scientific; and the physical and the digital in order to inspire and enable research, learning, experimentation, and play for a diverse community of users.']
	trial_matrix = tf_vectorizer.transform(trial)
	print(str(type(trial_matrix)))
	print(str(trial_matrix))
	trial_affinity = lda.transform(trial_matrix)
	print(str(trial_foo[0]))

	# comparing our results to the universe of known data
	print('=================================================================')
	print('=================================================================')
	print('=================================================================')
	print('Comparing results to known data')
	dummy = [trial_foo[0]]
	print(str(dummy))

	trial_pairwise = pairwise.cosine_distances(universe, dummy)
	print(str(type(trial_pairwise)))
	print(str(trial_pairwise))

	print('Finished!')
