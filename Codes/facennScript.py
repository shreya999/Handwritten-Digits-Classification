
'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy as sp

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    sig_aj = 1.0 / (1.0 + np.exp(-z))
    return sig_aj

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


    # Your code here
    #
    #
    #
    #
    #
	
	
	
    temp = np.ones((training_data.shape[0], training_data.shape[1] + 1))
    temp[:, :-1] = training_data
    training_data = temp
    print("after adding bias")
    		                                # constatnt feature
    a1= np.dot(training_data,w1.T)					#linear response
    z1= sigmoid(a1)							#activation const

    print("after sigmoid")
    temp = np.ones((z1.shape[0], z1.shape[1] + 1))
    temp[:, :-1] = z1
    z1 = temp
    print("after adding hidden bias")
    a2= np.dot(z1, w2.T)
    z2 = sigmoid(a2)
	
    y = np.zeros((training_data.shape[0],n_class))
    print("after output")

    obj_val = 0

    for i in range(0, training_data.shape[0]):
        y[i][int(training_label[i])] = 1
    obj_val = np.sum(y * np.log(z2) + ((1 - y) * np.log(1 - z2)))
    obj_val = -(obj_val / training_data.shape[0])
    obj_val = obj_val + ((lambdaval / (2 * training_data.shape[0])) * (np.sum(np.square(w1)) + np.sum(np.square(w2))))
    print("after obj_val",obj_val)
    #obj_grad

    delta = z2 - y
    grad_w2 = np.dot(np.transpose(delta),z1)
    grad_w2 = (grad_w2 + lambdaval*w2)/training_data.shape[0]

    grad_w1_a = (1 - z1) * z1
    grad_w1_b = np.dot(delta, w2)
    grad_w1_c = grad_w1_a * grad_w1_b
    grad_w1 = np.dot(np.transpose(grad_w1_c), training_data)

    grad_w1 = grad_w1[0:-1,:]

    grad_w1 = (grad_w1 +lambdaval*w1)/training_data.shape[0]
	
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    print("after obj_grad")
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here

    temp = np.ones((data.shape[0], data.shape[1] + 1))
    temp[:, :-1] = data
    data = temp
    print(data.shape)
    		                                # constatnt feature
    a1= np.dot(data,w1.T)					#linear response
    z1= sigmoid(a1)							#activation const

    temp = np.ones((z1.shape[0], z1.shape[1] + 1))
    temp[:, :-1] = z1
    z1 = temp
	
    a2= np.dot(z1, w2.T)
    z2 = sigmoid(a2)

    labels=[z2.argmax(axis=1)]
	#labels= z2.argmax()
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
