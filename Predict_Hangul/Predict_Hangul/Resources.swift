//
//  Resources.swift
//  Predict_Hangul
//
//  Created by pook on 1/10/20.
//  Copyright © 2020 pookjw. All rights reserved.
//

import Foundation
import TensorFlow
import Python

let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")

struct HangulBatch {
    let features: Tensor<Float32>
    let labels: Tensor<Int32>
}

let hiddenSize = 128

struct HangulModel: Layer {
    var layer1 = Dense<Float32>(inputSize: 784, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float32>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float32>(inputSize: hiddenSize, outputSize: 8)
    
    mutating func train(epoch_count: Int, train_data: Dataset<HangulBatch>, validation_data: Dataset<HangulBatch>) -> ([Float], [Float], [Float], [Float]){ // returns trainAccuracyResults, trainLossResults, validationAccuracyResults, validationLossResults
        var trainAccuracyResults: [Float] = []
        var trainLossResults: [Float] = []
        var validationAccuracyResults: [Float] = []
        var validationLossResults: [Float] = []

        for epoch in 1...epoch_count {
            var epochLoss: Float = 0
            var epochAccuracy: Float = 0
            var epochValLoss: Float = 0
            var epochValAccuracy: Float = 0
            var trainBatchCount: Int = 0
            var validationBatchCount: Int = 0
            
            // Train
            for batch in trainDataset{
                let (loss, grad) = self.valueWithGradient { (model: HangulModel) -> Tensor<Float32> in
                    let logits = model(batch.features)
                    return softmaxCrossEntropy(logits: logits, labels: batch.labels)
                }
                optimizer.update(&self, along: grad)
                
                let logits = self(batch.features)
                epochAccuracy += accuracy(preconditions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
                epochLoss += loss.scalarized()
                trainBatchCount += 1
            }
            
            // Validate
            for batch in validationDataset{
                let (loss, grad) = self.valueWithGradient { (model: HangulModel) -> Tensor<Float32> in
                    let logits = self(batch.features)
                    return softmaxCrossEntropy(logits: logits, labels: batch.labels)
                }
                //optimizer.update(&model, along: grad)
                
                let logits = self(batch.features)
                epochValAccuracy += accuracy(preconditions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
                epochValLoss += loss.scalarized()
                validationBatchCount += 1
            }
            
            epochAccuracy /= Float(trainBatchCount)
            epochLoss /= Float(trainBatchCount)
            
            epochValAccuracy /= Float(validationBatchCount)
            epochValLoss /= Float(validationBatchCount)
            
            trainAccuracyResults.append(epochAccuracy)
            trainLossResults.append(epochLoss)
            
            validationAccuracyResults.append(epochValAccuracy)
            validationLossResults.append(epochValLoss)
            
            print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy), Val Loss: \(epochValLoss), Val Accuracy: \(epochValAccuracy)")
        }
        
        return (trainAccuracyResults, trainLossResults, validationAccuracyResults, validationLossResults)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float32>) -> Tensor<Float32> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

func accuracy(preconditions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float32>(preconditions .== truths).mean().scalarized()
}

// extension from https://github.com/tensorflow/swift/blob/master/docs/site/tutorials/TutorialDatasetCSVAPI.swift

extension HangulBatch: TensorGroup {
    public static var _typeList: [TensorDataType] = [
        Float32.tensorFlowDataType,
        Int32.tensorFlowDataType
    ]
    public static var _unknownShapeList: [TensorShape?] = [nil, nil]
    public var _tensorHandles: [_AnyTensorHandle] {
        fatalError("unimplemented")
    }
    public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
        address!.advanced(by: 0).initialize(to: features.handle._cTensorHandle)
        address!.advanced(by: 1).initialize(to: labels.handle._cTensorHandle)
    }
    public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
        features = Tensor(handle: TensorHandle(_owning: tensorHandles!.advanced(by: 0).pointee))
        labels = Tensor(handle: TensorHandle(_owning: tensorHandles!.advanced(by: 1).pointee))
    }
    public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
        fatalError("unimplemented")
    }
}

extension Sequence where Element == HangulBatch {
    var first: HangulBatch? {
        return first(where: { _ in true })
    }
}
