#  Predict_Hangul

Predict handwritten-Hangul image using [Swift for TensorFlow](https://www.tensorflow.org/swift). [Hangul](https://en.wikipedia.org/wiki/Hangul) is the Korean alphabet. This tutorial uses  Hangul PHD08 Dataset for training by ChonBuk National University.

I used Xcode 11.3, [Swift for TensorFlow Development Snapshot (12/23/19)](https://github.com/tensorflow/swift/blob/master/Installation.md) and Python 3.6.8. Before proceeding this tutorial, I recommend following [Model training walkthrough](https://www.tensorflow.org/swift/tutorials/model_training_walkthrough) with my [pookjw/Practice_Swift_for_TensorFlow/Tutorial](https://github.com/pookjw/Practice_Swift_for_TensorFlow/tree/master/Tutorial) project.

## Prepare Training, Validation Dataset

Download Hangul PHD08 Dataset from ChonBuk National University. ([DropBox Link](https://www.dropbox.com/s/69cwkkqt4m1xl55/phd08.alz?dl=0), [Source](http://cv.jbnu.ac.kr/index.php?mid=notice&document_srl=189)) This Dataset is compressed as `.alz`. (I don't know why they did this.) To extract this you have to use `.al`z extractor like [Bandizip](https://en.bandisoft.com/bandizip/) (Windows version is free, but macOS version is paid).

In Hangul PHD08 Dataset, there are total 2,350 hangul image files like below.

```
$ ls phd08
가.txt            끓.txt        복.txt        죗.txt
개.txt            끔.txt        볶.txt        죙.txt
갸.txt            끕.txt        본.txt        죡.txt

...

끈.txt        볐.txt        죈.txt        힙.txt
끊.txt        병.txt        죌.txt        힛.txt
끌.txt        볕.txt        죔.txt        힝.txt
끎.txt        볜.txt        죕.txt
```

To train these data, we have to convert to `.npy` file using [sungjunyoung/phd08-conversion](https://github.com/sungjunyoung/phd08-conversion). phd08-conversion uses [imresize](https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.imresize.html) from `scipy`, but imresize is deprecated so you have to install SciPy 1.1.0.

`$ python3 -m pip uninstall scipy`

`$ python3 -m pip install scipy==1.1.0`

I will train **라, 호, 댜, 밟, 쟈, 꺅, 갠, 아** alphabets. Put these PHD08 images to `8_images` directory, and covert to 28x28 `npy` using phd08-conversion.

```
$ python3 phd08-conversion-master/phd08_to_npy.py --data_dir=8_images --width=28 --height=28
Namespace(batch_size=2, data_dir='8_images', gaussian_sigma=0.3, height=28, one_hot=False, width=28)
INFO:: converting 라.txt...
INFO:: converting 호.txt...
  FILE_SAVED:: filename : phd08_npy_results/phd08_data_1
INFO:: converting 댜.txt...
INFO:: converting 밟.txt...
  FILE_SAVED:: filename : phd08_npy_results/phd08_data_2
INFO:: converting 자.txt...
INFO:: converting 꺅.txt...
  FILE_SAVED:: filename : phd08_npy_results/phd08_data_3
INFO:: converting 갠.txt...
INFO:: converting 아.txt...
  FILE_SAVED:: filename : phd08_npy_results/phd08_data_4
INFO:: all files converted to npy file, results in phd08_npy_results
```

Then you can see `phd08_npy_results` on current directory.

```
$ ls phd08_npy_results
phd08_data_1.npy    phd08_data_3.npy    phd08_labels_1.npy    phd08_labels_3.npy
phd08_data_2.npy    phd08_data_4.npy    phd08_labels_2.npy    phd08_labels_4.npy
```

Each data npy files have `(4374, 28, 28)` shape. I will convert to `(4374, 784)` shape. Also each data and labels npy has two Hangul, one of them has 2,187 data. I will seperate 2,187 data to **1,800 data for training**, **387 data for validation**, and concatenate 4 npy files (8 Hangul) using `train_valid_npy.py` located in this project folder. Then I get `(14400, 784)` npy shape for training, `(3096, 784)` npy shape for validation. (1800*8=14400, 387*8=3096)

`$ python3 train_valid_npy.py`

After running `train_valid_npy.py`, you get `train_images.npy` (shaped as (14400, 784)), `train_labels.npy` (shaped as (14400, )), `validation_images.npy` (shaped as `(3096, 784`) and `validation_labels.npy` (shaped as (3096, )). I will train and validate these 4 npy files.

## Setup resources

Open `Predict_Hangul.xcodeproj`. Then you can see `main.swift` in project. In `main.swift`, import `TesnforFlow`, `Python` and load 4 npy files.

```swift
import Foundation
import TensorFlow
import Python
PythonLibrary.useVersion(3, 6)

let train_images = np.load("train_images.npy")
let train_labels = np.load("train_labels.npy")
let validation_images = np.load("validation_images.npy")
let validation_labels = np.load("validation_labels.npy")

let batchSize = 30

let trainDataset: Dataset<HangulBatch> = Dataset(elements: HangulBatch(features: Tensor<Float32>(numpy: train_images)!, labels: Tensor<Int32>(numpy: train_labels)!)).batched(batchSize)
let validationDataset: Dataset<HangulBatch> = Dataset(elements: HangulBatch(features: Tensor<Float32>(numpy: validation_images)!, labels: Tensor<Int32>(numpy: validation_labels)!)).batched(batchSize)
```

and set model, uses [Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam).

```swift
var model = HangulModel()
let optimizer = Adam(for: model)
```

## Train & Validate data

In `main.swift`

```swift
let epochCount = 20
var (trainAccuracyResults, trainLossResults, validationAccuracyResults, validationLossResults) = model.train(epoch_count: epochCount, train_data: trainDataset, validation_data: validationDataset)
```

Output below...

```
Epoch 1: Loss: 1.0767591, Accuracy: 0.85701376, Val Loss: 10.437824, Val Accuracy: 0.25576922
Epoch 2: Loss: 0.5076943, Accuracy: 0.90493035, Val Loss: 10.141166, Val Accuracy: 0.3330128
Epoch 3: Loss: 0.37693927, Accuracy: 0.92444444, Val Loss: 13.46332, Val Accuracy: 0.27051285
Epoch 4: Loss: 0.31091782, Accuracy: 0.95659715, Val Loss: 18.20225, Val Accuracy: 0.30608976
Epoch 5: Loss: 0.29377884, Accuracy: 0.9456944, Val Loss: 16.89025, Val Accuracy: 0.35705125
Epoch 6: Loss: 0.27239788, Accuracy: 0.95381933, Val Loss: 14.041648, Val Accuracy: 0.3451923
Epoch 7: Loss: 0.39004025, Accuracy: 0.9266665, Val Loss: 17.601622, Val Accuracy: 0.37692308
Epoch 8: Loss: 0.26391047, Accuracy: 0.95743066, Val Loss: 16.951427, Val Accuracy: 0.37724358
Epoch 9: Loss: 0.21627265, Accuracy: 0.9731251, Val Loss: 12.754482, Val Accuracy: 0.42051283
Epoch 10: Loss: 0.15818566, Accuracy: 0.9674999, Val Loss: 13.926802, Val Accuracy: 0.4583333
Epoch 11: Loss: 0.17731221, Accuracy: 0.9807638, Val Loss: 10.809947, Val Accuracy: 0.4358974
Epoch 12: Loss: 0.13462782, Accuracy: 0.98708326, Val Loss: 8.581307, Val Accuracy: 0.524359
Epoch 13: Loss: 0.08824263, Accuracy: 0.9875693, Val Loss: 12.346243, Val Accuracy: 0.42852563
Epoch 14: Loss: 0.19230917, Accuracy: 0.979236, Val Loss: 13.890122, Val Accuracy: 0.45576924
Epoch 15: Loss: 0.052777957, Accuracy: 0.99187493, Val Loss: 8.85447, Val Accuracy: 0.47692305
Epoch 16: Loss: 0.056914553, Accuracy: 0.992014, Val Loss: 8.249362, Val Accuracy: 0.5310897
Epoch 17: Loss: 0.056548968, Accuracy: 0.99326366, Val Loss: 6.5158787, Val Accuracy: 0.5519231
Epoch 18: Loss: 0.042444058, Accuracy: 0.99423605, Val Loss: 4.8365607, Val Accuracy: 0.6096154
Epoch 19: Loss: 0.017931798, Accuracy: 0.9966667, Val Loss: 0.32514527, Val Accuracy: 0.89198714
Epoch 20: Loss: 0.004365056, Accuracy: 0.99965274, Val Loss: 0.0076729646, Val Accuracy: 0.99871796
```

If you want to visualize accuracy, loss graph, use `plt` on `main.swift`

```swift
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
```

![Graph](https://live.staticflickr.com/65535/49366104277_ccb046c947_o.png)

## Prepare test data

I prepared **라 (0.jpeg), 밟 (4.jpeg), 꺅 (5.jpeg)** handwritten-image for test data. We have to convert these image to npy using `test_npy.py` located in this project.

`$ python3 test_npy.py`

After runnung `test_npy.py`, you get `test_images.npy` (shaped as (2, 784)), `train_labels.npy` (shaped as (2, )). I will predict this npy file. In `main.swift`...

```swift
let test_images = np.load("/Users/pook/Desktop/test_images.npy")
let test_labels = np.load("/Users/pook/Desktop/test_labels.npy")

let testDataset: Dataset<HangulBatch> = Dataset(elements: HangulBatch(features: Tensor<Float32>(numpy: test_images)!, labels: Tensor<Int32>(numpy: test_labels)!)).batched(batchSize)

let firstTestBatch = testDataset.first!
let firstTestBatchPredictions = model(firstTestBatch.features)

print("Predicted: \(firstTestBatchPredictions.argmax(squeezingAxis: 1))")
print("Answer: \(firstTestBatch.labels)")
```

Output below...

```
Predicted: [4, 5, 5]
Answer: [4, 5, 0]
```
