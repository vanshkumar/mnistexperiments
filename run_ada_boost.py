import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys

from read_data import read_in_data
from ada_boost import ada_boost_classifier

def main():
	train_subset = 10000
	test_subset = 2000

	train_img, train_lbl, test_img, test_lbl = read_in_data(train_subset, test_subset)

	# for i in range(train_subset):
	# 	plt.imshow(train_img[i,:,:], cmap=cm.Greys_r)
	# 	plt.show()

	# blah = train_img[0,:,:]
	# plt.imshow(blah, cmap = cm.Greys_r)
	# plt.show()

	train_img = train_img/255.0
	train_lbl = (train_lbl==5).astype(int) * 2 - 1

	print np.sum(train_lbl==1)

	test_img = test_img/255.0
	test_lbl = (test_lbl==5).astype(int) * 2 - 1

	print np.sum(test_lbl==1)

	feature_options = []
	for i in range(train_img.shape[1]*train_img.shape[2]):
		feature_options.append("raw_pixel_" + str(i))

	ada = ada_boost_classifier(train_img, train_lbl, feature_options)
	ada.train(25)
	ada.print_parameters()

	print "train"
	train_pred = ada.predict(train_img, ada.alpha, ada.thresholds, ada.signs, ada.feat_indices)
	train_error = np.sum(train_pred != train_lbl)
	print train_error/float(train_lbl.shape[0])

	print "test"
	test_pred = ada.predict(test_img, ada.alpha, ada.thresholds, ada.signs, ada.feat_indices)
	test_error = np.sum(test_pred != test_lbl)
	print test_error/float(test_lbl.shape[0])

if __name__ == "__main__":
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	main()