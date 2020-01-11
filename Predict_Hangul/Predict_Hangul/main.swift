//
//  main.swift
//  Predict_Hangul
//
//  Created by pook on 1/9/20.
//  Copyright © 2020 pookjw. All rights reserved.
//

import Foundation
import TensorFlow
import Python
PythonLibrary.useVersion(3, 6)

// Setup

//let class_names = ["라", "호", "댜", "쟈", "밟", "꺅", "갠", "아"]

let train_images = np.load("/Users/pook/Desktop/train_images.npy")
let train_labels = np.load("/Users/pook/Desktop/train_labels.npy")
let validation_images = np.load("/Users/pook/Desktop/validation_images.npy")
let validation_labels = np.load("/Users/pook/Desktop/validation_labels.npy")

let batchSize = 30

let trainDataset: Dataset<HangulBatch> = Dataset(elements: HangulBatch(features: Tensor<Float32>(numpy: train_images)!, labels: Tensor<Int32>(numpy: train_labels)!)).batched(batchSize)
let validationDataset: Dataset<HangulBatch> = Dataset(elements: HangulBatch(features: Tensor<Float32>(numpy: validation_images)!, labels: Tensor<Int32>(numpy: validation_labels)!)).batched(batchSize)

var model = HangulModel()
let optimizer = Adam(for: model)

// Training & Validation

let epochCount = 20
var (trainAccuracyResults, trainLossResults, validationAccuracyResults, validationLossResults) = model.train(epoch_count: epochCount, train_data: trainDataset, validation_data: validationDataset)


// Plot Training & Validation Result

plt.figure()

plt.subplot(221)
plt.plot(trainAccuracyResults)
plt.title("Accuracy")
plt.xlabel("Epoch")

plt.subplot(222)
plt.plot(trainLossResults)
plt.title("Loss")
plt.xlabel("Epoch")

plt.subplot(223)
plt.plot(validationAccuracyResults)
plt.title("Val Accuracy")
plt.xlabel("Epoch")

plt.subplot(224)
plt.plot(validationLossResults)
plt.title("Val Accuracy")
plt.xlabel("Epoch")

plt.show()

// Prediction

let test_images = np.load("/Users/pook/Desktop/test_images.npy")
let test_labels = np.load("/Users/pook/Desktop/test_labels.npy")

let testDataset: Dataset<HangulBatch> = Dataset(elements: HangulBatch(features: Tensor<Float32>(numpy: test_images)!, labels: Tensor<Int32>(numpy: test_labels)!)).batched(batchSize)

let firstTestBatch = testDataset.first!
let firstTestBatchPredictions = model(firstTestBatch.features)

print("Predicted: \(firstTestBatchPredictions.argmax(squeezingAxis: 1))")
print("Answer: \(firstTestBatch.labels)")
