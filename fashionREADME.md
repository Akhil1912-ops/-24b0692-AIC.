1.Setup steps for local execution.
.setup the environment
.install requirements(requirements.txt)

Detailed documentation of analysis, experiments

import the required libraries
load the csv files and split them into pixels and labels
3.make them standartize((224,224) and 3channels(rgb) image) as inputs to resnet50
4.split the traing data into 2 parts1 for validation and 1 for training
5.build a model 
6.mention its loss funtion and optimizers
7.define the new  head sutable for resnet50 
8.train the head(fc) and evaluvate head
9.load the layer4 and finetune it and evaluvate it
10.now test the model 
11.save the model



EXTERNAL RESOURCES USED
youtube videos-learned about each step (setting up model,loss funtions,optimizers)
internet-learned about how this neuron network works and how that structure works,
chatgtp-used for code syntex and setting parameters like learning rate and epochs


Notes on error handling and troubleshooting.
most of the errors where due to syntex .took help from internet and chatgtp



