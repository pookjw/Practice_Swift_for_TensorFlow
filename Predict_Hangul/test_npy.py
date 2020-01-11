import numpy as np
from scipy.misc import imresize
from scipy.misc import imread

test_image_1 = imread('/Users/pook/Desktop/4.jpeg', flatten=True, mode='I')
test_image_1 = imresize(test_image_1, [28, 28])
test_image_1 = ((test_image_1 / 255.0) - 1 ) * (-1)
test_image_1 = test_image_1.reshape(1, 28, 28)
test_image_1 = test_image_1.reshape(1, 784)

test_image_2 = imread('/Users/pook/Desktop/5.jpeg', flatten=True, mode='I')
test_image_2 = imresize(test_image_2, [28, 28])
test_image_2 = ((test_image_2 / 255.0) - 1 ) * (-1)
test_image_2 = test_image_2.reshape(1, 28, 28)
test_image_2 = test_image_2.reshape(1, 784)

test_image_3 = imread('/Users/pook/Desktop/0.jpeg', flatten=True, mode='I')
test_image_3 = imresize(test_image_3, [28, 28])
test_image_3 = ((test_image_3 / 255.0) - 1 ) * (-1)
test_image_3 = test_image_3.reshape(1, 28, 28)
test_image_3 = test_image_3.reshape(1, 784)

test_images = np.concatenate((test_image_1, test_image_2, test_image_3), axis=0).astype(np.float32)
test_labels = np.array([4, 5, 0]).astype(np.int32)

np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
