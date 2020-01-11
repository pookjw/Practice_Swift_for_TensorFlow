import numpy as np

temp_1 = np.load('/Users/pook/Desktop/phd08_npy_results/phd08_data_1.npy', mmap_mode='r')
temp_2 = np.load('/Users/pook/Desktop/phd08_npy_results/phd08_data_2.npy', mmap_mode='r')
temp_3 = np.load('/Users/pook/Desktop/phd08_npy_results/phd08_data_3.npy', mmap_mode='r')
temp_4 = np.load('/Users/pook/Desktop/phd08_npy_results/phd08_data_4.npy', mmap_mode='r')
temp_5 = np.load('/Users/pook/Desktop/phd08_npy_results/phd08_labels_1.npy', mmap_mode='r')
temp_6 = np.load('/Users/pook/Desktop/phd08_npy_results/phd08_labels_2.npy', mmap_mode='r')
temp_7 = np.load('/Users/pook/Desktop/phd08_npy_results/phd08_labels_3.npy', mmap_mode='r')
temp_8 = np.load('/Users/pook/Desktop/phd08_npy_results/phd08_labels_3.npy', mmap_mode='r')

temp_1 = np.reshape(temp_1, (4374, 784))
temp_2 = np.reshape(temp_2, (4374, 784))
temp_3 = np.reshape(temp_3, (4374, 784))
temp_4 = np.reshape(temp_4, (4374, 784))

train_images = np.concatenate((temp_1[:1800], temp_1[2187:3987], temp_2[:1800], temp_2[2187:3987], temp_3[:1800], temp_3[2187:3987], temp_4[:1800], temp_4[2187:3987]), axis=0) / 255.0
train_labels = np.concatenate((temp_5[:1800], temp_5[2187:3987], temp_6[:1800], temp_6[2187:3987], temp_7[:1800], temp_7[2187:3987], temp_8[:1800], temp_8[2187:3987]), axis=0).astype(np.int32)

validation_images = np.concatenate((temp_1[1800:2187], temp_1[3987:4374], temp_2[1800:2187], temp_2[3987:4374], temp_3[1800:2187], temp_3[3987:4374], temp_4[1800:2187], temp_4[3987:4374]), axis=0) / 255.0
validation_labels = np.concatenate((temp_5[1800:2187], temp_5[3987:4374], temp_6[1800:2187], temp_6[3987:4374], temp_7[1800:2187], temp_7[3987:4374], temp_8[1800:2187], temp_8[3987:4374]), axis=0).astype(np.int32)

np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)

np.save('validation_images.npy', validation_images)
np.save('validation_labels.npy', validation_labels)
