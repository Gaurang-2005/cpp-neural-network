# Neural Network From Scratch in Pure C++

This project is a complete implementation of a feed-forward neural network written entirely in **pure C++**, without using any machine learning or linear algebra libraries.

The goal was to deeply understand how neural networks actually work internally — from neuron math to backpropagation — before using high-level frameworks.

---

## What This Project Is

A fully working neural network framework built from scratch that includes:

* Custom **Neuron**, **Layer**, and **NeuralNetwork** classes
* Forward propagation
* Backpropagation with manual gradient computation
* Multiple activation functions
* Mean Squared Error loss
* Flexible architecture (any number of layers)

No external libraries were used for ML or matrix math.

---

## Why I Built This

I enjoy working with low-level systems and am currently learning C++ while trying to take it to an advanced level.

Instead of relying on machine learning libraries, I wanted to build a neural network completely from scratch to understand:

* How weights are updated
* How gradients flow through layers
* How activation functions affect learning
* How loss functions guide optimization

Building everything manually helped me understand neural networks much more deeply.

---

## Implemented Concepts

### Neural Network Components

* Neuron class with weights, bias, and activation
* Layer class managing multiple neurons
* NeuralNetwork class managing layers and training

### Activation Functions

* ReLU
* Sigmoid
* Linear

Each activation includes its derivative for backpropagation.

---

## Training Algorithm

The network uses:

* Forward propagation to compute predictions
* Mean Squared Error loss
* Backpropagation to compute gradients
* Gradient descent to update weights and biases

All derivatives are computed manually.

---

## Architecture Example

Example network you can build with this code:

```
Input → Hidden Layer → Hidden Layer → Output
```

Because the framework is flexible, you can create networks for:

* XOR problems
* Function approximation
* Classification tasks
* Image datasets like MNIST

---

## How To Compile

Using g++:

```
g++ neuralNetwork.cpp -O2 -std=c++20 -o nn
./nn
```

You can then write your own `main()` to test different datasets or architectures.

---

## What I Learned

Through this project I learned:

* How neural networks actually compute outputs
* Why initialization matters
* How gradients propagate through layers
* How activation choice affects convergence

Understanding these details makes using ML libraries much more meaningful.

---

## Acknowledgement

The neuralNetwork class and core training logic were written entirely by me and tested by me.

ChatGPT was used as a guide for explanations, debugging ideas, and structuring some example main programs.

---

## If You Find This Useful

Feel free to use the code, experiment with it, or suggest improvements.

Understanding ML from scratch is one of the best ways to really learn it.
