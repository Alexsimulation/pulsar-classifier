# Neural Network that classifies pulsar candidates

## Overview
The challenge is to predict, using 8 various measurements as inputs, if the candidate star really is a pulsar. The dataset contains pulsar candidates, with 1639 positive samples and 16259 negative samples.

## Results
The pre-trained network present in this repository took about 1000 iterations of the whole training set to train, the set being divided in 23 batches. The *pretrained.txt* file contains a readable list of the weights and biases of each layer, and the *pretrained.net* file can be loaded in the main.py program. The performance of the classifier is presented below:
- Right on any types: 92.6%
- Right on negatives: 92.8%
- Right on positives: 88.4%

## Network structure
It's a simple multi-layer perceptron classifier. All layers are fully connected layers, and the nodes use the sigmoid activation function. The network learns using backpropagated batch gradients, using the log loss function.
0. Inputs : 8 nodes
1. Hidden : 16 nodes
2. Hidden : 8 nodes
3. Output : 1 node

## Sources
- Pulsar Dataset HTRU2: https://www.kaggle.com/charitarth/pulsar-dataset-htru2
- Backpropagation implementation with least squares: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
