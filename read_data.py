import struct
import sys

from array import array

TRAIN_IMAGES = 'Train/train-images.idx3-ubyte'
TRAIN_LABELS = 'Train/train-labels.idx1-ubyte'

TEST_IMAGES = 'Test/t10k-images.idx3-ubyte'
TEST_LABELS = 'Test/t10k-labels.idx1-ubyte'

train_subset = 1000
test_subset = 200

def load(images_file, labels_file, subset):
	image_data = []
	labels     = []

	with open(images_file, 'rb') as im:
		magic, size, rows, cols = struct.unpack(">IIII", im.read(16))

		if magic != 2051:
			print "error with magic # for images"
			sys.exit(0)

		image_data = array("B", im.read(subset*4*rows*cols))

	with open(labels_file, 'rb') as lbl:
		magic, size = struct.unpack(">II", lbl.read(8))

		if magic != 2049:
			print "error with magic # for labels"
			sys.exit(0)

		labels = array("B", lbl.read(subset*4))

	images = []
	for i in xrange(subset):
		images.append(image_data[i*rows*cols : (i+1)*rows*cols])

	return images, labels


def read_in_data():
	train_img, train_lbl = load(TRAIN_IMAGES, TRAIN_LABELS, train_subset)
	test_img, test_lbl = load(TEST_IMAGES, TEST_LABELS, test_subset)

	return train_img, train_lbl, test_img, test_lbl