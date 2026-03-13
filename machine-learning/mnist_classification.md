# Understanding the MNIST Handwritten Digit Classifier

## What Is This Notebook About?

This notebook teaches you how to build a program that **looks at pictures of handwritten digits (0-9) and figures out which number it is**. Think of it like teaching a computer to read your handwriting, except we start simple and progressively build smarter models.

The notebook builds **five different models**, each one more sophisticated than the last, so you can see how adding complexity improves accuracy.

---

## The Big Picture: How Is This Different from C++?

In C++, you write explicit rules: `if (x > 5) { do something; }`. **Machine learning flips this around.** Instead of writing rules, you show the computer thousands of examples and let it *figure out the rules on its own*. You never write code that says "if there's a loop in the top-left, it's probably a 9." The model discovers patterns like that by itself.

| C++ Programming            | Machine Learning                      |
| -------------------------- | ------------------------------------- |
| You write the rules        | The computer learns the rules         |
| Logic is explicit in code  | Logic is hidden in learned "weights"  |
| Works on exact conditions  | Works on patterns and probabilities   |

---

## A Quick Python Primer for C++ Programmers

Before we dive into the code, here's a crash course on the Python you'll see in this notebook. If you've written C++, you already understand most of the concepts. The syntax is just different.

### Variables: no types required

In C++, you declare a variable with its type:

```cpp
int x = 5;
double accuracy = 0.95;
std::string name = "mnist";
```

In Python, you just assign:

```python
x = 5
accuracy = 0.95
name = "mnist"
```

Python figures out the type automatically. No semicolons, no type declarations. You can even reassign `x = "hello"` later and Python won't complain (for better or worse).

### Lists instead of arrays

C++ arrays have a fixed type and size. Python **lists** are flexible and can hold anything:

```python
my_list = [1, 2, 3, 4, 5]       # like int arr[] = {1,2,3,4,5};
my_list[0]                      # access first element (same as C++)
```

You'll also see **tuples**, which use parentheses: `(28, 28)`. Tuples are like lists that can't be modified after creation. They're used frequently to describe shapes and dimensions.

### Importing libraries

In C++, you use `#include` to bring in headers:

```cpp
#include <iostream>
#include <cmath>
```

In Python, you use `import`:

```python
import numpy as np             # import numpy, nickname it "np"
from tensorflow import keras   # import just "keras" from tensorflow
```

The `as np` part gives the library a short nickname so you can write `np.mean()` instead of `numpy.mean()` everywhere.

### Objects and the dot (`.`) operator

You've seen `std::string` in C++. When you create a string and call a method on it:

```cpp
std::string s = "hello";
int len = s.length();    // calling a method with the dot operator
```

Python works the same way. Almost everything in Python is an **object** (an instance of a class), and you call methods on it with the dot:

```python
s = "hello"
upper_s = s.upper()      # calling a method with the dot operator
```

In this notebook, you'll see chains of dots like `keras.layers.Dense(...)`. This is just navigating through nested objects: `keras` is a library, `layers` is a module inside it, and `Dense` is a class inside that module. Think of it like nested namespaces in C++ (`std::chrono::seconds`).

### Functions and named arguments

In C++, you call a function by position:

```cpp
pow(2.0, 3.0);  // base=2, exponent=3
```

Python lets you use **named arguments** (also called keyword arguments), which makes code much more readable:

```python
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

You could write this positionally, but names make it clear which value goes where. This is especially helpful when functions take many parameters.

### No braces, no semicolons

Python uses **indentation** (whitespace) instead of `{ }` to define blocks:

```python
# Python               # C++ equivalent:
for i in range(5):     # for (int i = 0; i < 5; i++) {
    print(i)           #     cout << i << endl;
                       # }
```

If your indentation is wrong, Python will give you an error. This forces the code to look clean, but it can trip you up at first.

### f-strings (formatted output)

Instead of `cout << "Value: " << x << endl;`, Python uses **f-strings**:

```python
x = 42
print(f"The value is {x}")         # prints: The value is 42
print(f"Accuracy: {0.9094:.2%}")   # prints: Accuracy: 90.94%
```

The `f` before the quote means "look for `{...}` inside and replace them with values." The `:.2%` part is formatting: show as a percentage with 2 decimal places.

---

## Cell 0: Loading and Visualizing the Data

```python
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

plt.imshow(x_train[0], cmap="gray")
plt.title(f"Value: {y_train[0]}")
```

### What's happening here?

**The MNIST dataset** is a famous collection of 70,000 images of handwritten digits. It's the "Hello, World!" of machine learning. Each image is 28x28 pixels in grayscale.

The first three lines import the libraries we need:

- **`keras`** (from TensorFlow): the machine learning framework that lets us define and train models.
- **`numpy`** (nicknamed `np`): a math library for working with arrays of numbers efficiently. Think of it as a supercharged replacement for C++ arrays that can do math on entire arrays at once.
- **`matplotlib.pyplot`** (nicknamed `plt`): a plotting library for displaying images and graphs.

The data is split into two groups:

- **Training set** (`x_train`, `y_train`): 60,000 images the model learns from
- **Test set** (`x_test`, `y_test`): 10,000 images the model has *never seen*, used to check how well it actually learned

Think of it like studying for an exam: the training set is your textbook, and the test set is the actual exam. You'd be suspicious if someone only tested you on problems you'd already memorized!

**`x_train`** contains the images (the pixel data, i.e., the *input*).
**`y_train`** contains the labels (the correct answer, e.g., "this image is a 5").

Notice the line `(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()`. In Python, you can **unpack** a function's return value into multiple variables at once. The function returns two tuples (one for training, one for testing), and we unpack each tuple into its `x` and `y` parts in a single line. In C++, you'd need separate assignments.

### Why divide by 255?

Each pixel is originally a number from 0 (black) to 255 (white). Dividing by 255 **rescales everything to be between 0 and 1**. Neural networks learn much faster and more reliably when input values are small. This is called **normalization**. It's like converting inches to a 0-1 scale so the math stays manageable.

Notice that `x_train / 255.0` divides *every element* in the array at once. In C++, you'd need a `for` loop. NumPy lets you do math on entire arrays in one expression. This is called **vectorized operations**, and it's one of the biggest reasons Python (with NumPy) is popular for data science.

### The image display

The last two lines display the first training image so you can see what the data looks like: a 28x28 grayscale picture of a digit. `plt.imshow()` renders an array as an image, and `plt.title()` adds a label above it showing the correct digit.

---

## Cell 1: Model 1, Random Guessing (The Baseline)

```python
random_preds = np.random.randint(0, 10, size=y_test.shape)
random_accuracy = np.mean(random_preds == y_test)
print(f"Random Guessing Accuracy: {random_accuracy:.2%}")
```

**Result: ~10% accuracy**

### Why do we do this?

Before building anything fancy, we establish a **baseline**, the dumbest possible approach. If you randomly guess a digit between 0 and 9, you'd expect to be right about 1 out of 10 times (10%). Any real model needs to beat this, or it's literally worse than guessing.

### How the code works

- `np.random.randint(0, 10, ...)` generates random integers from 0 to 9, one guess per test image.
- `random_preds == y_test` compares each guess to the true answer, producing `True` (1) or `False` (0) for every element. Again, this is a vectorized operation: no `for` loop needed.
- `np.mean(...)` averages those 1s and 0s, giving us the fraction that were correct.

This is conceptually identical to what you'd do in C++: loop through two arrays, count matches, divide by total. Python/NumPy just lets you write it in one line.

---

## Cell 2: Model 2, The Naive Model (No Hidden Layers)

```python
naive_model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation="softmax")
])

naive_model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
naive_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

**Result: ~91% accuracy**

### What is a neural network layer?

Think of a **layer** as a step in a pipeline. Data goes in one side, gets transformed, and comes out the other. In C++ terms, it's like a function that takes an array as input and returns a new array as output, except the function's behavior is defined by numbers (called **weights**) that the computer adjusts during training.

### A note on the Python syntax

`keras.Sequential([...])` creates a new `Sequential` **object** (an instance of the `Sequential` class). The `[...]` inside is a Python list of layer objects. Each layer (like `keras.layers.Dense(...)`) is also an object being created. So you're building a list of layer objects and passing that list to the `Sequential` constructor.

When we later call `naive_model.compile(...)` and `naive_model.fit(...)`, we're calling **methods** on the model object, just like calling `s.length()` on a `std::string` in C++.

### Breaking down each part

**`Input(shape=(28, 28))`**: Tells the model "expect a 28x28 grid of numbers." The `shape` parameter is a tuple describing the dimensions. This is like declaring the type of your function parameter in C++.

**`Flatten()`**: Converts the 2D image (28 rows x 28 columns) into a single 1D list of 784 numbers. Think of reading a book page: instead of a grid of characters, you read them left-to-right, top-to-bottom into one long string. Neural network layers need flat input.

**`Dense(10, activation="softmax")`**: This is where the magic happens. "Dense" means **every one of the 784 input values is connected to every one of the 10 output values**. Each connection has a "weight" (a number). The model multiplies inputs by weights, adds them up, and produces 10 numbers, one for each digit (0-9).

**Softmax** converts those 10 raw numbers into **probabilities** that add up to 1.0. For example: `[0.01, 0.02, 0.05, 0.80, 0.01, 0.03, 0.02, 0.03, 0.02, 0.01]` means the model is 80% sure it's a "3".

### The three steps after defining the model

#### Step 1: Compile

Sets up *how* the model will learn.

- **Optimizer (`"sgd"`)**: Stochastic Gradient Descent. Imagine you're blindfolded on a hilly landscape and want to find the lowest valley. SGD takes small steps downhill based on the slope under your feet. The "stochastic" part means it estimates the slope from a small random sample of data (faster but slightly noisy).
- **Loss function (`"sparse_categorical_crossentropy"`)**: Measures *how wrong* the model is. "Cross-entropy" means: if the true answer is "5" and the model only gave "5" a 20% probability, the loss is high. If it gave "5" a 99% probability, the loss is low. The model tries to *minimize* this number. "Sparse" just means labels are plain integers (5) rather than one-hot vectors ([0,0,0,0,0,1,0,0,0,0]).
- **Metrics (`["accuracy"]`)**: What we display during training, i.e., the percentage of correct predictions.

#### Step 2: Fit (Train)

Runs through the entire training set 5 times (`epochs=5`). Each pass adjusts the weights slightly to reduce the loss. It's like doing 5 practice exams and improving your study strategy after each one.

#### Step 3: Evaluate

Tests the final model on the test set and prints the accuracy.

### Why only ~91%?

This model has no "thinking" layers in the middle. It directly maps pixels to digits. It's like trying to recognize a face by looking at each freckle individually without ever considering how features relate to each other.

---

## Cell 3: Model 3, Simple Neural Network (One Hidden Layer)

```python
simple_model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),   # <-- NEW hidden layer
    keras.layers.Dense(10, activation="softmax")
])
```

**Result: ~94% accuracy**

### What changed?

We added a **hidden layer** with 128 nodes between the input and output. This is the key idea behind neural networks: intermediate layers can learn **abstract features** (things like "is there a curve here?" or "is there a straight vertical line?") that are useful for the final classification.

### What is ReLU?

**ReLU** stands for "Rectified Linear Unit." It sounds fancy, but it's the simplest useful activation function:

```text
relu(x) = max(0, x)
```

If the input is negative, output 0. If positive, pass it through unchanged. In C++:

```cpp
double relu(double x) {
    return x > 0 ? x : 0;
}
```

That's it! So why is this trivial function so important? Without it (or something like it), stacking layers would be pointless. Multiple linear transformations in a row collapse into a single linear transformation. ReLU adds **non-linearity**, which is what lets the network learn complex, curved decision boundaries instead of just straight lines.

### Why 128 nodes?

128 is a common choice (it's a power of 2, which is efficient for GPUs). It's not magic; you could try 64, 256, or any number. More nodes means more capacity to learn patterns, but also more computation and risk of **overfitting** (memorizing the training data instead of learning general rules).

---

## Cell 4: Model 4, Multilayer Perceptron (Two Hidden Layers)

```python
mlp_model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),     # <-- second hidden layer
    keras.layers.Dense(10, activation="softmax")
])
mlp_model.compile(optimizer="adam", ...)
```

**Result: ~97.8% accuracy**

### What changed?

Two things:

1. **A second hidden layer** (64 nodes). The first layer might learn low-level features (edges, strokes), and the second layer combines those into higher-level features (loops, corners, intersections). It's like how you might first notice individual pen strokes, then recognize them as parts of a letter.

2. **The optimizer switched from `"sgd"` to `"adam"`**. Adam is a smarter version of SGD. It adapts its step size for each weight individually and uses momentum (like a ball rolling downhill that builds up speed). This is why the accuracy jumped so much: **Adam converges faster and more reliably**, especially on problems like this.

### What is a "perceptron"?

A perceptron is a single artificial neuron. It takes inputs, multiplies each by a weight, sums them up, and passes the result through an activation function. A "Multilayer Perceptron" (MLP) is just multiple layers of these neurons stacked together. It's the classic neural network architecture.

---

## Cell 5: Model 5, Convolutional Neural Network (CNN)

```python
cnn_model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
```

**Result: ~98.4% accuracy**

### Why is this the best model for images?

All the previous models used `Flatten()` as their first real step. They treated the image as a long list of 784 unrelated numbers. This **throws away spatial information**. The model doesn't know that pixel (5,5) is *next to* pixel (5,6). It's like scrambling the pixels of a photo and trying to recognize what's in it.

**CNNs preserve spatial structure.** Instead of looking at all 784 pixels at once, they slide small windows (called **filters** or **kernels**) across the image.

### How Conv2D works

`Conv2D(32, (3, 3))` creates 32 different 3x3 filters. Each filter is a tiny 3x3 grid of learnable weights. The filter slides across the image one pixel at a time:

```text
Image patch:     Filter:         Result:
[1, 0, 1]       [1, 0, 1]
[0, 1, 0]   x   [0, 1, 0]   =   sum of element-wise products
[1, 0, 1]       [1, 0, 1]
```

Each filter learns to detect a specific visual pattern. One might detect horizontal edges, another vertical edges, another curves, etc. The model figures out *which* patterns are useful during training.

Think of it like this: instead of looking at an entire page at once, you scan it with a magnifying glass, noting interesting features at each position.

### How MaxPooling2D works

`MaxPooling2D((2, 2))` shrinks the data by taking each 2x2 block and keeping only the **maximum** value:

```text
[4, 1]
[2, 7]  =  7
```

This does two things:

1. **Reduces computation** (the data is now half the size in each dimension).
2. **Adds some position tolerance.** The exact pixel location of a feature matters less, which helps because people don't always write digits in the exact same spot.

### How the data shape changes through the CNN

It helps to trace the shape of the data as it flows through each layer:

1. **Input**: `(28, 28, 1)`, a 28x28 image with 1 grayscale channel.
2. **After `Conv2D(32, (3, 3))`**: The 3x3 filter can't center on the outermost edge pixels (it would hang off the image), so the output shrinks by 2 in each dimension: `28 - 3 + 1 = 26`. With 32 filters, the output shape is `(26, 26, 32)`.
3. **After `MaxPooling2D((2, 2))`**: Each dimension is halved (every 2x2 block becomes one value): `26 / 2 = 13`. The output shape is `(13, 13, 32)`.
4. **After `Flatten()`**: All three dimensions are multiplied together into a single flat list: `13 * 13 * 32 = 5,408`. The output shape is `(5408,)`.

So by the time the data reaches the `Dense(64)` layer, each image has been reduced from 784 raw pixels to 5,408 *learned feature values*. That number is larger than 784 because the 32 filters each produced their own 13x13 feature map, but the information is much more meaningful: instead of raw brightness values, these numbers represent "how strongly did filter #N activate at position (row, col)?"

### Why the `(28, 28, 1)` input shape?

The `1` represents the **color channel**. Grayscale images have 1 channel. Color images (RGB) would have 3. Conv2D expects this channel dimension to be explicit.

### Why is it slower?

The convolution operation is more computationally expensive than simple matrix multiplication. Notice in the output that each epoch takes ~30 seconds vs ~5 seconds for the simpler models. But the accuracy gain is worth it for image tasks.

---

## Summary: The Progression

| Model        | Architecture              | Accuracy | Key Idea                                      |
| ------------ | ------------------------- | -------- | --------------------------------------------- |
| Random Guess | No learning               | ~10%     | Baseline: how well can you do by chance?       |
| Naive        | Input -> Output           | ~91%     | Direct pixel-to-digit mapping                 |
| Simple NN    | Input -> 128 -> Output    | ~94%     | One hidden layer learns intermediate features  |
| MLP          | Input -> 128 -> 64 -> Out | ~97.8%   | Deeper network + Adam optimizer               |
| CNN          | Conv -> Pool -> 64 -> Out | ~98.4%   | Respects spatial structure of images           |

The key takeaway: **more sophisticated architectures that respect the structure of the data lead to better results.** Images have spatial patterns, and CNNs are designed to find them.

---

## Glossary for C++ Programmers

- **Model**: Like a function with parameters that get tuned automatically.
- **Weights**: The adjustable numbers inside the model (like variables that the training process sets for you).
- **Training**: Running data through the model repeatedly, comparing its output to the correct answer, and nudging the weights to do better next time.
- **Epoch**: One complete pass through the training dataset.
- **Loss**: A single number measuring how wrong the model is (lower is better).
- **Optimizer**: The algorithm that decides how to adjust weights to reduce the loss.
- **Activation function**: A simple function applied to each node's output to introduce non-linearity (ReLU, softmax).
- **Overfitting**: When the model memorizes the training data but fails on new data (like memorizing answers instead of understanding concepts).
- **Vectorized operation**: Performing math on an entire array at once instead of looping element by element. This is what NumPy provides and why Python is practical for data science despite being slower than C++ in general.
