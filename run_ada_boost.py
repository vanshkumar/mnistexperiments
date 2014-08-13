import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys

from read_data import read_in_data
from ada_boost import ada_boost_classifier

def main():
	train_subset = 10
	test_subset = 2

	train_img, train_lbl, test_img, test_lbl = read_in_data(train_subset, test_subset)

	# for i in range(train_subset):
	# 	plt.imshow(train_img[i,:,:], cmap=cm.Greys_r)
	# 	plt.show()

	# blah = train_img[0,:,:]
	# plt.imshow(blah, cmap = cm.Greys_r)
	# plt.show()

	train_img = train_img/255.0
	train_lbl = (train_lbl>4).astype(int) * 2 - 1

	print train_lbl

	feature_options = []
	for i in range(train_img.shape[1]*train_img.shape[2]):
		feature_options.append("raw_pixel_" + str(i))

	ada = ada_boost_classifier(train_img, train_lbl, feature_options)
	ada.train(10)
	ada.print_parameters()

if __name__ == "__main__":
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	main()