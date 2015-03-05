
from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import check_grad
from scipy.io import loadmat
from math import sqrt
from math import log
import time

countNn = 0

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
    
    
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    sigmoid_of_z = (1.0 / (1.0 + np.exp(-z)))
    return sigmoid_of_z
    
    
def preprocess():

    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    start = time.clock()
    print "Preprocessing started."
    input_file = '/home/hrishikesh/Projects/ML/Project1/mnist_all.mat'
    #input_file = 'E:\ML\PA1\Project 1\mnist_all.mat'
    mat = loadmat(input_file) #loads the MAT object as a Dictionary
    
    '''Split data into temporary arrays - one for training data and 
       other for test data. Add labels to respective lists. 
       Labels are in the range 0-9.'''

    train_temp = []
    test_temp = []
    train_label_temp = []
    test_label_temp = []

    for item in mat:
        if "test" in item:
            for i in range(0, len(mat.get(item))):
                label = [0 for x in range(0,10)]
                label[int(item[-1])] = 1
                test_temp.append(mat.get(item)[i] / 255)
                test_label_temp.append(label)
        elif "train" in item:
            for i in range(0, len(mat.get(item))):
                label = [0 for x in range(0,10)]
                label[int(item[-1])] = 1
                train_temp.append(mat.get(item)[i] / 255)
                train_label_temp.append(label)

    '''If a column contains the same elements add it to the to_delete list'''
    to_delete = []
    for i in range(0, len(train_temp[0])):
        is_equal = True
        for j in range(0, len(train_temp) - 1):
            if train_temp[j][i] != train_temp[j+1][i]:
                is_equal = False
                break
        if is_equal:
            to_delete.append(i)

    '''Delete the columns from the training data. Size reduced from 784 to 717'''
    count = 0
    for i in to_delete:
        train_temp = np.delete(train_temp, i - count, 1)
        test_temp = np.delete(test_temp, i - count, 1)
        count += 1

    '''Get the size of the input and generate an array of size = size_of_input with 
       elements in a random permutation in range(0, size_of_input).
       Example - size_of_input = 5. aperm = [3,1,2,0,4]'''

    size_of_input = range(train_temp.shape[0])
    aperm = np.random.permutation(size_of_input)

    '''Use the aperm array to split training data into training and validation data.
       Do the same for labels'''

    train_data = np.array(train_temp[aperm[100:1000],:])
    train_label = np.array([train_label_temp[x] for x in aperm[100:1000]])
    validation_data = np.array(train_temp[aperm[0:100],:])
    validation_label = np.array([train_label_temp[x] for x in aperm[0:100]])
    test_data = np.array(test_temp)
    test_label = np.array(test_label_temp)
    end = time.clock()
    print "Preprocessing completed"
    print "Time taken: %f" % (end - start)    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
  
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    global countNn
    countNn += 1
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    grad_w1 = [[0.0 for j in range(0, w1.shape[1])] for i in range(0, n_hidden)]
    grad_w2 = [[0.0 for j in range(0, n_hidden + 1)] for i in range(0, n_class)]

    number_of_training_data = training_data.shape[0]
    eq_5 = np.array([])
    eq_15 = get_eq_15(w1, w2, lambdaval, number_of_training_data)

    for i in range(0, number_of_training_data):    
        
        # d attributes in example (x). d = 717
        example = training_data[i]

        # Add bias node to make it d + 1 = 718
        example = np.append(example, 0)
        
        # Calculate output of hidden nodes. Calculated using w1T.x
        # Gives a 50x1 array. Add bias node to give 51x1 array
        output_of_hidden = np.array([])
        for j in range(0, n_hidden):
            output_of_hidden = np.append(output_of_hidden, sigmoid(np.dot(example, w1[j])))
        output_of_hidden = np.append(output_of_hidden, 1)
        output_of_hidden = output_of_hidden.reshape(1, output_of_hidden.size)


        # Calculate output of classification. Gives a 10x1 array.
        output_of_class = np.array([])
        for l in range(0, n_class):
            o = sigmoid(np.dot(output_of_hidden, w2[l]))
            output_of_class = np.append(output_of_class, o)
        output_of_class = output_of_class.reshape(10, 1)

        temp = get_eq_5(w1, w2, np.transpose(output_of_class), training_label, n_class, i)
        eq_5 = np.append(eq_5, temp)

        # Calculate gradient of w2 - shape - 10x51
        delta = np.array([])
        delta = np.subtract(output_of_class, train_label[i].reshape(10,1))
        grad_w2 = np.add(grad_w2, np.dot(delta, output_of_hidden))
       
        output_of_hidden_new = np.array([])
        delta_l_w2 = np.dot(np.transpose(delta), w2)
        for j in range(0, n_hidden):
            x = (1 - output_of_hidden[0][j]) * output_of_hidden[0][j] * delta_l_w2[0][j]
            output_of_hidden_new = np.append(output_of_hidden_new, x)
        output_of_hidden_new = output_of_hidden_new.reshape(50, 1)
        example = example.reshape(1, 718)
        grad_w1 = np.add(grad_w1, np.dot(output_of_hidden_new, example))

    '''Divide each element in the gradient matrix by number of examples'''

    obj_val = (-1 * np.sum(eq_5) / number_of_training_data) + eq_15
    print obj_val
    for i in range(0, len(grad_w2)):
        for j in range(0, len(grad_w2[0])):
            grad_w2[i][j] += (lambdaval * w2[i][j])
            grad_w2[i][j] /= number_of_training_data

    for i in range(0, len(grad_w1)):
        for j in range(0, len(grad_w1[0])):
            grad_w1[i][j] += (lambdaval * w1[i][j])
            grad_w1[i][j] /= number_of_training_data

    grad_w1 = np.array(grad_w1)
    grad_w2 = np.array(grad_w2)

    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)


    return (obj_val,obj_grad)


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = []
    #Your code here
    for i in range(0, data.shape[0]):    
        
        # d attributes in example (x). d = 717
        example = data[i]

        # Add bias node to make it d + 1 = 718
        example = np.append(example, 1)
        
        # Calculate output of hidden nodes. Calculated using w1T.x
        # Gives a 50x1 array. Add bias node to give 51x1 array
        output_of_hidden = np.array([])
        for j in range(0, w1.shape[0]):
            z = sigmoid(np.dot(example, w1[j]))
            output_of_hidden = np.append(output_of_hidden, z)
        output_of_hidden = np.append(output_of_hidden, 1)
        
        # Calculate output of classification. Gives a 10x1 array.
        output_of_class = np.array([])
        for l in range(0, w2.shape[0]):
            b = 0
            for j in range(0,  len(output_of_hidden)):
                b += output_of_hidden[j] * w2[l][j]
            o = sigmoid(b)
            output_of_class = np.append(output_of_class, o)
        labels.append(output_of_class)
    
    labels = np.argmax(labels, 1)
    return labels


def get_eq_15(w1, w2, lambdaval, n):
    return 0.0 if lambdaval == 0 else ((np.sum(np.square(w1)) + np.sum(np.square(w2))) * lambdaval) / (2 * n)


def get_eq_5(w1, w2, output_of_class, training_label, n_class, i):
    temp = training_label[i]
    temp = temp.reshape(10,1)
    x = np.dot(np.log(output_of_class), temp)
    y = np.dot(np.log(1 - output_of_class), 1 - temp)
    return (x + y)[0][0]


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);


# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.1;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

start = time.clock()
print "\nMinimize started."

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

print "Minimize finished."
end = time.clock()
print "Time taken: %f" % (end-start)
print "\nNumber of iterations: %d" % countNn

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset
train_label = np.argmax(train_label, 1)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset
validation_label = np.argmax(validation_label, 1)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
test_label = np.argmax(test_label, 1)
print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')