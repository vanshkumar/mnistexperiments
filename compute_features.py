"""
Example usage:

feature_selection = features_class(train_img)
feature_selection.add_raw_pixel_features()

features = feature_selection.get_features()

"""

import numpy as np
import sys

class features_class:
	def __init__(self, data):
		self.features = None
		self.data = data

	def _add_feature(self, to_add):
		if self.features is None:
			self.features = to_add
		else:
			self.features = np.concatenate((self.features, to_add), axis=1)

	def add_raw_pixel_features(self):
		(num_examples, rows, cols) = self.data.shape
		pixels = np.reshape(self.data, (num_examples, rows*cols))

		self._add_feature(pixels)

	def add_one_raw_pixel(self, i):
		(num_examples, rows, cols) = self.data.shape
		pixels = np.reshape(self.data, (num_examples, rows*cols))
		pixel = pixels[:, i]
		pixel = np.reshape(pixel, (pixel.shape[0], 1))

		self._add_feature(pixel)

	def add_list_of_features(self, feature_list):
		for feat in feature_list:
			if "raw_pixel_" in feat:
				num = int(feat[feat.rfind("_")+1:])
				self.add_one_raw_pixel(num)
			else:
				print "The feature name you gave, " + feat + \
					  ", is not supported"
				sys.exit(0)

	def get_features(self):
		return self.features