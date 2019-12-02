# Neural Network
---

> Results of the implementation could be seen in notebooks session, where i compare my NN with Keras.

![NeuralNetwork](https://raw.githubusercontent.com/rdenadai/AI-Study-Notebooks/master/images/nn.png)

## What's implemented:
 
 - FeedForward
 - Backprogapation

### Layers:

 - Dense (Fully Connected)
 - Dropout

### Activations:

 - ReLU
 - Sigmoid
 - Softmax

### Loss:

 - Cross Entropy

---
## Dropout

The Dropout layer serves as regularization layer... it turn on and off random units in the network in each pass of training, reducing overfit.

Bellow is a scheme of how Dropout works... dropout should be only activate during training.

<img src="https://raw.githubusercontent.com/rdenadai/AI-Study-Notebooks/master/images/dropout.png" width="500px"/>