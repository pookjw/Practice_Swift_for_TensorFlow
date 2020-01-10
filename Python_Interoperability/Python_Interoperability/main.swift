//
//  main.swift
//  Python_Interoperability
//
//  Created by pook on 1/9/20.
//  Copyright © 2020 pookjw. All rights reserved.
//

import Foundation
import TensorFlow
import Python

PythonLibrary.useVersion(3, 8)
print(Python.version)

// Convert standard Swift types to Python.
let pythonInt: PythonObject = 1
let pythonFloat: PythonObject = 3.0
let pythonString: PythonObject = "Hello Python!"
let pythonRange: PythonObject = PythonObject(5..<10)
let pythonArray: PythonObject = [1, 2, 3, 4]
let pythonDict: PythonObject = ["foo": [0], "bar": [1, 2, 3]]

// Perform standard operations on Python objects.
print(pythonInt + pythonFloat)
print(pythonString[0..<6])
print(pythonString[PythonObject(0..<6)])
print(pythonRange)
print(pythonArray[2])
print(pythonDict["bar"])

// Convert Python objects back to Swift.
let int = Int(pythonInt)!
let float = Float(pythonFloat)!
let string = String(pythonString)!
let range = Range<Int>(pythonRange)!
let array: [Int] = Array(pythonArray)!
let dict: [String: [Int]] = Dictionary(pythonDict)!

// Perform standard operations.
// Outputs are the same as Python!
print(Float(int) + float)
print(string.prefix(6))
print(range)
print(array[2])
print(dict["bar"]!)

/*
 PythonObject defines conformances to many standard Swift protocols:

 - Equatable
 - Comparable
 - Hashable
 - SignedNumeric
 - Strideable
 - MutableCollection
 - All of the ExpressibleBy_Literal protocols
 */

let one: PythonObject = 1
print( one == one )
print( one < one )
print( one + one )

let array_2: PythonObject = [1, 2, 3]
for (i, x) in array_2.enumerated() {
    print(i, x)
}

let pythonTuple = Python.tuple([1, 2, 3])
print(pythonTuple, Python.len(pythonTuple))

// Convert to Swift.
let tuple = pythonTuple.tuple3
print(tuple)

// `Python.builtins` is a dictionary of all Python builtins.
_ = Python.builtins

// Try some Python builtins.
print(Python.type(1))
print(Python.len([1, 2, 3]))
print(Python.sum([1, 2, 3]))

let np = Python.import("numpy")
print(np)
let zeros = np.ones([2, 3])
print(zeros)

let maybeModule = try? Python.attemptImport("nonexistent_module")
print(maybeModule)

let numpyArray = np.ones([4], dtype: np.float32)
print("Swift Type:", type(of: numpyArray))
print("Python Type:", Python.type(numpyArray))
print(numpyArray.shape)

// Examples of converting `numpy.ndarray` to Swift types.
let array_3: [Float] = Array(numpy: numpyArray)!
let shapedArray = ShapedArray<Float>(numpy: numpyArray)!
let tensor = Tensor<Float>(numpy: numpyArray)!

// Examples of converting Swift types to `numpy.ndarray`.
print(array_3.makeNumpyArray())
print(shapedArray.makeNumpyArray())
print(tensor.makeNumpyArray())

// Examples with different dtypes.
let doubleArray: [Double] = Array(numpy: np.ones([3], dtype: np.float))!
let intTensor = Tensor<Int32>(numpy: np.ones([2, 3], dtype: np.int32))!

let plt = Python.import("matplotlib.pyplot")

let time = np.arange(0, 10, 0.01)
let amplitude = np.exp(-0.1 * time)
let position = amplitude * np.sin(3 * time)

plt.figure(figsize: [15, 10])

plt.plot(time, position)
plt.plot(time, amplitude)
plt.plot(time, -amplitude)

plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Oscillations")

plt.show()
