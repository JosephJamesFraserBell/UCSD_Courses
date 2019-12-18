import torch.nn.init
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from tqdm import tqdm
from scipy.stats import truncnorm
import torch.nn.functional as torch_functional
import matplotlib
import numpy as np
import os
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
# Change path as required
path = r"C:\Users\josep\OneDrive\Desktop\UCSD_Courses\CSE252A\hw5"

def read(dataset = "training", datatype='images'):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    
    if(datatype=='images'):
        get_data = lambda idx: img[idx]
    elif(datatype=='labels'):
        get_data = lambda idx: lbl[idx]

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_data(i)
        
# example linear classifier - input connected to output
# you can take this as an example to learn how to extend DNN class
# class LinearClassifier(DNN):
#     def __init__(self, in_features=28*28, classes=10):
#         super(LinearClassifier, self).__init__()
#         # in_features=28*28
#         self.weight1 = weight_variable((classes, in_features))
#         self.bias1 = bias_variable((classes))
    
#     def forward(self, x):
#         # linear operation
#         y_pred = torch.addmm(self.bias1, x.view(list(x.size())[0], -1), self.weight1.t())
#         return y_pred
    
trainData=np.array(list(read('training','images')))
trainData=np.float32(np.expand_dims(trainData,-1))/255
trainData=trainData.transpose((0,3,1,2))
trainLabels=np.int32(np.array(list(read('training','labels'))))

testData=np.array(list(read('testing','images')))
testData=np.float32(np.expand_dims(testData,-1))/255
testData=testData.transpose((0,3,1,2))
testLabels=np.int32(np.array(list(read('testing','labels'))))
def DataBatch(data, label, batchsize, shuffle=True):
    n = data.shape[0]
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize : min(n,(i+1)*batchsize)]
        yield data[inds], label[inds]


class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        dtype = torch.float
        device = torch.device("cpu")


    def forward(self, x): # the classes that inherit DNN override this method with their own forward method 
        pass
    
    def train_net(self, trainData, trainLabels, epochs=1, batchSize=50):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = 3e-4)
        
        for epoch in range(epochs):
            self.train()  # set netowrk in training mode
            for i, (data,labels) in enumerate(DataBatch(trainData, trainLabels, batchSize, shuffle=True)):
                data = Variable(torch.FloatTensor(data))
                labels = Variable(torch.LongTensor(labels))
                
                # YOUR CODE HERE------------------------------------------------
                # Train the model using the optimizer and the batch data
                y_pred = self.forward(data)
                loss = criterion(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #------------------------------------------------------------------
                #-----End of your code, don't change anything else here------------
                
            self.eval()  # set network in evaluation mode
            print ('Epoch:%d Accuracy: %f'%(epoch+1, test(testData, testLabels, self))) 
    
    def __call__(self, x):
        inputs = Variable(torch.FloatTensor(x))
        prediction = self.forward(inputs)
        return np.argmax(prediction.data.cpu().numpy(), 1)

# helper function to get weight variable
def weight_variable(shape):
    initial = torch.Tensor(truncnorm.rvs(-1/0.01, 1/0.01, scale=0.01, size=shape))
    return Parameter(initial, requires_grad=True)

# helper function to get bias variable
def bias_variable(shape):
    initial = torch.Tensor(np.ones(shape)*0.1)
    return Parameter(initial, requires_grad=True)

def test(testData, testLabels, classifier):
    batchsize=50
    correct=0.
    for data,label in DataBatch(testData,testLabels,batchsize,shuffle=False):
        prediction = classifier(data)
        correct += np.sum(prediction==label)

        plt.imshow(data[0].reshape((28,28)), cmap='gray')
        print(prediction.shape)
        plt.title(prediction)
        plt.show()
    return correct/testData.shape[0]*100


def conv2d(x, W, stride):
    # x: input
    # W: weights (out, in, kH, kW)
    
    return F.conv2d(x, W, stride=stride, padding=2)

# Defining a Convolutional Neural Network
class CNNClassifer(DNN):
    def __init__(self, classes=10, n=5):
        super(CNNClassifer, self).__init__()
        """ ==========
        YOUR CODE HERE
        ========== """
        
        # dimensions for convolution were calculated by formula in lecture 18 notes
        # dimensions for everything else were found by guess and check and just filling in
        # the required dimension to match previous layer output
        
        ## first layer
        self.weight1 = weight_variable((n, 1, n, n))
        self.bias1 = bias_variable((50,5,14,14)) 
        
        ## second layer
        self.weight2 = weight_variable((n*n, 5, n, n))
        self.bias2 = bias_variable((50,25,7,7))
        
        ## third layer
        self.weight3 = weight_variable((50,1225))
        self.bias3 = bias_variable((50))
        
        ## fourth layer
        self.weight4 = weight_variable((10,50))
        self.bias4 = bias_variable((10))
        
        dtype = torch.float
        device = torch.device("cpu")
        
        
    def forward(self, x):
        """ ==========
        YOUR CODE HERE
        ========== """

        y_pred = conv2d(x,self.weight1,2)
        
        y_pred = torch_functional.relu(y_pred) + self.bias1
        
        y_pred = conv2d(y_pred,self.weight2,2)
        
        y_pred = torch_functional.relu(y_pred) + self.bias2
        
        y_pred = y_pred.view(list(y_pred.size())[0], -1)
        
        y_pred = torch.addmm(self.bias3, y_pred, self.weight3.t())
        
        y_pred = torch_functional.relu(y_pred)
        
        
        y_pred = torch.addmm(self.bias4, y_pred.view(list(y_pred.size())[0], -1), self.weight4.t())
        
        return y_pred
    
cnnClassifer = CNNClassifer()
cnnClassifer.train_net(trainData, trainLabels, epochs=10)

