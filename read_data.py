import numpy as np
import struct
import sys

from array import array

TRAIN_IMAGES = 'Train/train-images.idx3-ubyte'
TRAIN_LABELS = 'Train/train-labels.idx1-ubyte'

TEST_IMAGES = 'Test/t10k-images.idx3-ubyte'
TEST_LABELS = 'Test/t10k-labels.idx1-ubyte'

def load(images_file, labels_file, subset):
	image_data = []
	labels     = []

	with open(images_file, 'rb') as im:
		magic, size, rows, cols = struct.unpack(">IIII", im.read(16))

		if magic != 2051:
			print "error with magic # for images"
			sys.exit(0)

		image_data = array("B", im.read())

	with open(labels_file, 'rb') as lbl:
		magic, size = struct.unpack(">II", lbl.read(8))

		if magic != 2049:
			print "error with magic # for labels"
			sys.exit(0)

		labels = array("B", lbl.read())

	images = []
	for i in xrange(subset):
		images.append(image_data[i*rows*cols : (i+1)*rows*cols])

	np_images = np.zeros((subset, rows, cols))
	for i in range(len(images)):
		img = np.array(images[i], dtype=float)
		img = np.reshape(img, (rows, cols))
		np_images[i] = img

	return np_images, np.array(labels[:subset], dtype=int)


def read_in_data(train_subset, test_subset):
	train_img, train_lbl = load(TRAIN_IMAGES, TRAIN_LABELS, train_subset)
	test_img, test_lbl = load(TEST_IMAGES, TEST_LABELS, test_subset)

	return train_img, train_lbl, test_img, test_lbl