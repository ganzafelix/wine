#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def euclidean_distance(a,b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))#problem might be here

def load_data(csv_filename):
    from pandas.io.parsers import read_csv
    filew = read_csv(csv_filename)
    file = filew.to_numpy()
    fl_new = np.zeros((len(file),11))
    
    for x in range(0,len(file)):
        
        row = file[x,0].split(';')
        fl_new[x,0] = row[0]
        fl_new[x,1] = row[1]
        fl_new[x,2] = row[2]
        fl_new[x,3] = row[3]
        fl_new[x,4] = row[4]
        fl_new[x,5] = row[5]
        fl_new[x,6] = row[6]
        fl_new[x,7] = row[7]
        fl_new[x,8] = row[8]
        fl_new[x,9] = row[9]
        fl_new[x,10] = row[10]
        
        
    #for loops
    
    return fl_new 
    
    """ 
    Returns a numpy ndarray in which each row repersents
    a wine and each column represents a measurement. There should be 11
    columns (the "quality" column cannot be used for classificaiton).
    """
    pass    
    
def split_data(dataset, ratio):
    r = int(len(dataset)*ratio)
    #I add a 1 to make sure that the ratio I want is full(since the splitting doesn't consider the last number)
    training_set = dataset[:r+1]
    testing_set = dataset[r+1:]
     
    
    
    """
    Return a (train, test) tuple of numpy ndarrays. 
    The ratio parameter determines how much of the data should be used for 
    training. For example, 0.9 means that the training portion should contain
    90% of the data. You do not have to randomize the rows. Make sure that 
    there is no overlap. 
    """
    return (training_set,testing_set)
    
def compute_centroid(data):
    
    return sum(data)/data.shape[0]
    """
    Returns a 1D array (a vector), representing the centroid of the data
    set. 
    """
    pass
    
def experiment(ww_train, rw_train, ww_test, rw_test):
    #create the centroids
    red = compute_centroid(rw_train)
    white = compute_centroid(ww_train)
    
    correct = 0
    incorrect = 0
    #predict
    for x in ww_test :
        
        distR = euclidean_distance(red,x)
        distW = euclidean_distance(white, x)
        
        if distR < distW:
            incorrect += 1
            
        if distR > distW:
            correct += 1
            
    for x in rw_test :
        
        distR = euclidean_distance(red,x)
        distW = euclidean_distance(white, x)
        
        if distR < distW:
            correct += 1
            
        if distR > distW:
            incorrect += 1
            
        
    total = correct + incorrect
    accuracy = correct/total
    
    print('The total number of predictions made is {}\n There was {} correct predictions\n The accuracy is {}'.format(total,correct,accuracy))
    
    return accuracy
    
    """
    Train a model on the training data by creating a centroid for each class.
    Then test the model on the test data. Prints the number of total 
    predictions and correct predictions. Returns the accuracy. 
    """
    pass
    
def cross_validation(ww_data, rw_data, k):
    #create a list of accuracies
    li_accu = []
    #find a number that is divisible by k and use the number to split the arrays into equal partitions
    
    ww_copy = np.copy(ww_data)
    rw_copy = np.copy(rw_data)
    
    red_li = np.array_split(rw_copy,k)
    white_li = np.array_split(ww_copy,k) 
    
   
    rw_training = []
    ww_training = []
    ww_add = []
    rw_testing = []
    ww_testing = []
    rw_add = []
    
    
    count_rw = 0
    count_ww = 0
    
    
    for arr in white_li:
        training = white_li
        ww_testing.append(arr)
        nw_train = np.delete(training,count_ww,0)
        count_ww += 1
        ww_training.append(nw_train)
        
    for idx in range(len(ww_training)):
        stck_ww = np.concatenate((ww_training[idx][0:9]), axis = 0)
        ww_add.append(stck_ww)
        
    for arr in red_li:
        training = red_li
        rw_testing.append(arr)
        nw_train = np.delete(training,count_rw,0)
        count_rw += 1
        rw_training.append(nw_train)
    
    for idx in range(len(rw_training)):
        stck_rw = np.concatenate((rw_training[idx][0:9]), axis = 0)
        rw_add.append(stck_rw)
        
    
            
    for i in range(len(ww_testing)):
        accu = experiment(ww_add[i], rw_add[i], ww_testing[i], rw_testing[i])
       
        li_accu.append(accu)
                     
    
    mean = sum(li_accu)/len(li_accu)

    """
    Perform k-fold crossvalidation on the data and print the accuracy for each
    fold. 
    """
    return mean

    
if __name__ == "__main__":
    
    ww_data = load_data('whitewine.csv')
    rw_data = load_data('redwine.csv')

    #Uncomment the following lines for step 2: 
    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    experiment(ww_train, rw_train, ww_test, rw_test)
    
    
    # Uncomment the following lines for step 3:
    k = 10
    acc = cross_validation(ww_data, rw_data,k)
    print("{}-fold cross-validation accuracy: {}".format(k,acc))
    
