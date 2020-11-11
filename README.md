# visual-search

[![emabello42](https://circleci.com/gh/emabello42/visual-search.svg?style=shield)](https://circleci.com/gh/emabello42/visual-search)

Visual Search is a service to find similar images giving an input image or text description. It uses a pretrained CNN for both extraction of features from the last convolutional layer and the last fully connected layer to classify the input image. The features are then used to find the most similar images, and the output can be used together with a softmax function to obtain a category.
