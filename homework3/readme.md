This is the third homework assignment of the course: Neural Network Priciples and Application (2018) in Shanghai Jiao Tong Univiersity.

The problem is described as following:

In this assignment, convolutional neural network (CNN) will be used to deal with multi-class classification problems. CNN is a class of deep, feed-forward artificial neural networks that has successfully been applied to analyzing visual imagery.

Two problems are given below. The dataset used in this homework is the MNIST
database (Modified National Institute of Standards and Technology database), which is
commonly used for training and testing in the field of machine learning. 

A) Solving the ten-class classification problem in the given dataset using feedforward
neural network. You need to finetune your network and only present your best
result.

Notice: You can either use tensorflow or other deep learning tools to solve this problem.
You can also build your network without using any deep learning tools, which is a better
option

B) Solving the ten-class classification problem using CNN.
(a) You need to implement LeNet-5 and use it to solve this problem.
(b) Compare the results and training time with problem 1.
(c) Visualize the deep features which can be extracted before feed-forward layers, and discuss the results. 

Here is our code. The dataset can be downloaded automatically the first time when you run the code. You need to run the file main.py with deterministing that the argument of '--layer_num' is equal to your model's layers number. Our program would generate the training/validation loss value and classification accuracy during training.
