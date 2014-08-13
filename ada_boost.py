import os.path
import numpy as np
import sys

from compute_features import features_class
from math import log

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class ada_boost_classifier():
	def __init__(self, data, labels, feature_options):
		self.data = data
		self.feat_options = feature_options
		self.labels = labels

		self.weight = np.ones((labels.shape[0],))/labels.shape[0]
		self.error = []
		self.thresholds = []
		self.feat_indices = []
		self.signs = []
		self.alpha = []

	def calculate_error(self, data, labels, weight, alpha, thresholds, signs, \
			feat_indices):
		predictions = self.predict(data, alpha, thresholds, signs, feat_indices)

		error = (predictions != labels).astype(float)
		error = error * weight

		# print "error being calculated"
		# print thresholds
		# print signs
		# print feat_indices
		# print predictions
		# print labels
		# print weight
		# print error

		error = np.abs(np.sum(error))

		# print error

		return error

	def find_best_threshold(self, data, labels, weight):
		best_threshold = -1
		final_sign = -1
		final_index = 0
		lowest_error = 10000000

		alpha = np.array([1])

		for feat_index in range(len(self.feat_options)):
			feat_error = lowest_error
			feat_threshold = -1
			feat_sign = -1

			for sign in (-1, 1):
				for thresh in range(-10, 10):
					error = self.calculate_error(data, labels, weight, \
						alpha, np.array([thresh]), np.array([sign]), \
						[feat_index])

					#print error

					if error < feat_error:
						feat_error = error
						feat_threshold = thresh
						feat_sign = sign

			#print "With pixel " + str(feat_index) + " we have error " + str(feat_error)

			if feat_error < lowest_error:
				lowest_error = feat_error
				best_threshold = feat_threshold
				final_sign = feat_sign
				final_index = feat_index

		return (lowest_error, best_threshold, final_sign, final_index)

	def new_classifier(self):
		error, threshold, sign, index = \
			self.find_best_threshold(self.data, self.labels, self.weight)

		error = error + 1e-15

		self.thresholds.append(threshold)
		self.error.append(error)
		self.signs.append(sign)
		self.alpha.append(1/2.0 * log((1-error)/error, 10))
		self.feat_indices.append(index)

		# Calculate the new distribution of weights
		predictions = self.predict(self.data, np.array([1]), threshold, sign, \
			[index])

		for i in range(len(self.weight)):
			wt = self.weight[i]
			pred = predictions[i]
			label = self.labels[i]

			if pred == label:
				self.weight[i] = wt / (2 * (1 - error))
			else:
				self.weight[i] = wt / (2 * error)

	def predict(self, data, alpha, thresholds, signs, feat_indices):
		feature_selector = features_class(data)
		feature_names = [self.feat_options[i] for i in feat_indices]
		feature_selector.add_list_of_features(feature_names)

		features = feature_selector.get_features()

		#print features.shape
		#print features

		weak_classifiers = np.zeros(features.shape)

		for i in range(features.shape[0]):
			signed = np.array(signs)*features[i,:]
			thresholded = signed - np.array(thresholds)
			weighted = np.array(alpha)*np.sign(thresholded)
			weak_classifiers[i, :] = weighted

		strong_classifier = np.sign(np.sum(weak_classifiers, axis=1))
		strong_classifier = np.transpose(strong_classifier)

		return strong_classifier

	def train(self, max_iter):
		for i in range(max_iter):
			self.new_classifier()

			if self.error[-1] < 0.01:
				break

	def print_parameters(self):
		print "Features: "
		print [self.feat_options[i] for i in self.feat_indices]

		print "Error: "
		print self.error

		print "Thresholds: "
		print self.thresholds

		print "Signs: "
		print self.signs

		print "Weights: "
		print self.alpha