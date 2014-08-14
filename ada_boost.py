import os.path
import numpy as np
import sys

from compute_features import features_class
from math import sqrt, log

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class ada_boost_classifier():
	def __init__(self, data, labels, feature_options):
		self.data = data
		self.feat_options = feature_options
		self.labels = labels

		self.weight = np.array([1.0]*labels.shape[0])/labels.shape[0]
		self.error = []
		self.thresholds = []
		self.feat_indices = []
		self.signs = []
		self.alpha = []

	def calculate_error(self, data, labels, weight, alpha, thresholds, signs, \
			feat_indices):
		predictions = self.predict(data, alpha, thresholds, signs, feat_indices)

		error = (predictions != labels).astype(float)
		error = np.sum(error * weight)

		# print "error being calculated"
		# print thresholds
		# print signs
		# print feat_indices
		# print predictions
		# print labels
		# print weight
		# print error

		return error

	def find_best_threshold(self, data, labels, weight):
		best_threshold = -1
		final_sign = -1
		final_index = 0

		lowest_error = 1000000
		if len(self.error) > 0:
			lowest_error = self.error[-1]

		for feat_index in range(len(self.feat_options)):
			feat_error = lowest_error
			feat_threshold = -1
			feat_sign = -1

			for sign in (-1, 1):
				for thresh in np.arange(-1, 1, 0.1):
					error = self.calculate_error(data, labels, weight, [1], \
						[thresh], [sign], [feat_index])

					if error < feat_error:
						feat_error = error
						feat_threshold = thresh
						feat_sign = sign

			# print "With pixel " + str(feat_index) + " we have error " + str(feat_error)

			if feat_error < lowest_error:
				print "Previous lowest error: " + str(lowest_error)
				print "New lowest error: " + str(feat_error)
				print "With feature index " + str(feat_index)
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

		x = sqrt((1 - error) / error)

		self.alpha.append(log(x, 10))
		self.feat_indices.append(index)

		print ''
		pred = self.predict(self.data, self.alpha, self.thresholds, self.signs, self.feat_indices)
		print "Non-weighted error: " + str(np.sum(pred != self.labels)/float(self.labels.shape[0]))
		print ''

		# Calculate the new distribution of weights
		predictions = self.predict(self.data, [1], [threshold], [sign], [index])

		for i in range(len(self.weight)):
			wt = self.weight[i]
			pred = predictions[i]
			label = self.labels[i]

			if pred == label:
				self.weight[i] = wt / x
			else:
				self.weight[i] = wt * x
		self.weight /= np.sum(self.weight)

	def predict(self, data, alpha, thresholds, signs, feat_indices):
		feature_selector = features_class(data)
		feature_names = [self.feat_options[i] for i in feat_indices]
		feature_selector.add_list_of_features(feature_names)

		features = feature_selector.get_features()

		# print "Feature indices: "
		# print feature_names

		# print "Predicting"

		# print features

		signed = np.multiply(features, np.array(signs))

		# print "Signs"
		# print np.array(signs)
		# print signed

		thresholded = np.subtract(signed, np.array(thresholds))

		# print "Threshold"
		# print np.array(thresholds)
		# print thresholded

		weighted = np.multiply(np.sign(thresholded), np.array(alpha))

		# print "Weighted"
		# print np.sign(thresholded)
		# print np.array(alpha)
		# print weighted

		strong_classifier = np.sign(np.sum(weighted, axis=1))
		strong_classifier = np.transpose(strong_classifier)

		return strong_classifier

	def train(self, max_iter):
		for i in range(max_iter):
			self.new_classifier()

			if i > 2 and (self.error[-1] - self.error[-2]) > 0.001:
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