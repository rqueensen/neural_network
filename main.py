import sys
import numpy as np
import csv
import scipy
from sklearn import svm, datasets
import random
import math
from sklearn.preprocessing import scale
import time

n_in = 784
n_hid = 200
n_out = 10

def main(argv):
    np.random.seed(6)
    random.seed(6)
    
    if argv[0] == "train":
        train()
    elif argv[0] == "kaggle":
        kaggle()
            
def train():
    #prints data from training in different conditions
    
    train_data = scipy.io.loadmat("train.mat")
    tr_data = train_data['train_images']
    labels = train_data['train_labels']
    labels = labels.flatten()

    data = []
    for i in range(60000):
        data.append(tr_data[:,:,i].flatten())
    tr_data = data
    
    #Shuffle
    z = list(zip(tr_data, labels))
    random.shuffle(z)
    tr_data, labels = zip(*z)
    tr_data = np.matrix(tr_data, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    
    #preprocess
    tr_data = scale(tr_data)
    tr_data = np.append(tr_data, np.zeros((60000,1)) + 1., axis=1)
    
    #Split into 50K training, 10K validation
    x_chunks = np.array_split(tr_data, 6)
    y_chunks = np.array_split(labels, 6)
    val_x = np.matrix(x_chunks[5])
    val_y = y_chunks[5]
    indicies = list(range(6))
    indicies.remove(5)
    training_x = np.take(x_chunks, indicies, axis=0)
    training_y = np.take(y_chunks, indicies, axis=0)
    train_x = np.matrix(np.concatenate(training_x, axis=0))
    train_y = np.array(np.concatenate(training_y, axis=0))
    

    for i in range(1, 11):
        V, W = train_neural_network(train_x, train_y, .01, i)
        val_predictions = predict(val_x, V, W)
        training_predictions = predict(train_x, V, W)
        
        print(i, error(val_predictions, val_y), error(training_predictions, train_y))
       
def train_neural_network(images, labels, learn_rate, epoch, cross=False):
    #Returns update weights after training over epoch epochs
    
    V, W = getWeights()
    
    iter = 0
    index = 0
    length = len(images)
    while (iter < epoch):
        x = images[index]
        y = np.zeros((10,1))
        y[labels[index]] = 1.
        
        #Forward
        h, z = forward(x, V, W)
        
        #Back Prop
        if cross:
            dJdz = np.divide(np.multiply(x, (1-y)), (1 - np.multiply(x, z))) - np.divide(np.multiply(y, np.log(x)), np.multiply(z, np.square(np.log(z))))
            dLdz = np.dot(dJdz,np.multiply(z, 1-z))
        else:
            dLdz = np.multiply(2*(z-y),np.multiply(z, 1-z))
        
        dLdW = np.dot(dLdz, h.T)
        
        dLdV = np.dot(np.multiply(1 - np.square(h), np.dot(dLdz.T, W).T), x)
        dLdV = np.delete(dLdV, [200], axis=0)
    
        #Update Weights
        W -= learn_rate * dLdW
        V -= learn_rate * dLdV

        #Update counters
        index += 1
        if index == length:
            iter += 1
            index = 0
            
    return V, W

def predict(images, V, W):
    #Returns a list of predicted labels for images based on the weights
    
    predictions = []
    for image in images:
        h, z = forward(image, V, W)
        predictions.append(np.argmax(z))
    return predictions
        
def error(predicted, actual):
    #Returns the classification error of predicted labels vs actual
    
    total = 0
    for i in range(len(actual)):
        if predicted[i] != actual[i]:
            total += 1
            
    return total / len(actual)
    
def sigmoid(g):
    return 1./(1. + np.exp(-g))
    
def forward(image, V, W):
    #performs the forward step, returns h and predictions z
    
    h = np.tanh(np.dot(V, image.T))
    h = np.append(h, [[1.]], axis=0)
    z = sigmoid(np.dot(W, h))
    
    return h, z
       
def cost(predicted, labels):
    return .5 * sum((labels - predicted)**2)
      
def getWeights():
    #Returns randomly generated weight matricies
    
    V = np.zeros((n_hid, n_in +1))
    W = np.zeros((n_out, n_hid +1))
    
    for i in range(n_hid):
        for j in range(n_in +1):
            V[i][j] = random.gauss(0., .01)
            
    for i in range(n_out):
        for j in range(n_hid +1):
            W[i][j] = random.gauss(0., .01)
            
    return V, W
  
def kaggle():
    #Produces kaggle output file
    
    train_data = scipy.io.loadmat("train.mat")
    tr_data = train_data['train_images']
    labels = train_data['train_labels']
    labels = labels.flatten()

    test_data = scipy.io.loadmat("test.mat")
    test_images = test_data["test_images"]

    data = []
    for i in range(60000):
        data.append(tr_data[:,:,i].flatten())
    tr_data = data
    
    data = []
    for i in range(10000):
        data.append(test_images[i].flatten())
    test_images = data
    
    #Shuffle
    z = list(zip(tr_data, labels))
    random.shuffle(z)
    tr_data, labels = zip(*z)
    tr_data = np.matrix(tr_data, dtype=np.float64)
    test_images = np.matrix(test_images, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    
    #preprocess
    tr_data = scale(tr_data)
    tr_data = np.append(tr_data, np.ones((60000,1)), axis=1)
    
    test_images = scale(test_images)
    test_images = np.append(test_images, np.ones((10000,1)), axis=1)
    
    x_chunks = np.array_split(tr_data, 6)
    test_chunks = np.array_split(test_images, 6)
    y_chunks = np.array_split(labels, 6)
    indicies = list(range(6))
    training_x = np.take(x_chunks, indicies, axis=0)
    testing_x = np.take(test_chunks, indicies, axis=0)
    training_y = np.take(y_chunks, indicies, axis=0)
    train_x = np.matrix(np.concatenate(training_x, axis=0))
    train_y = np.array(np.concatenate(training_y, axis=0))
    test_x = np.matrix(np.concatenate(testing_x, axis=0))
    
    t0 = time.time()
    V, W = train_neural_network(train_x, train_y, .01, 50)
    test_predictions = predict(test_x, V, W)
    
    
    with open("predict.csv", "w+") as predictions:
        swriter = csv.writer(predictions, delimiter=',', lineterminator="\n")
        swriter.writerow(["Id", "Category"])
        
        for i in range(len(test_predictions)):
            swriter.writerow([i+1, test_predictions[i]])
    
    print((time.time() - t0)/60)
    
if __name__ == "__main__":
    main(sys.argv[1:])