import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import pickle

def convert_labels_to_array_of_labels(file):

    """
    this function converts class labels from file y_train.txt file to numpy array of class labels

    Parameters:
    -----------
    :param data: ytrain.txt

    Returns:
    --------
    :return: returns array of class labels
    """

    lines=file.readlines()
    indices=[]
    for i in range(len(lines)):
        indices.append(lines[i].replace('\n',''))
    indices_array=np.array(indices)

    return indices_array

def array_to_one_hot_array(indices,depth,on_value,off_value,axis):

    """
    this function converts array of elements to one hot array

    Parameters:
    -----------

    :param indices: A Tensor of indices.
    :param depth:  A scalar defining the depth of the one hot dimension.
    :param on_value: A scalar defining the value to fill in output when indices[j] = i. (default: 1)
    :param off_value: A scalar defining the value to fill in output when indices[j] != i. (default: 0)
    :param axis: The axis to fill (default: -1, a new inner-most axis).

    Returns:
    --------
    :return:The one-hot tensor

    """

    one_hot_array=tf.one_hot(indices,depth,on_value,off_value,axis)
    return one_hot_array


def plot_class_distribution_bar(data, xlabel, ylabel, image_name):
    """
    this function takes list of classes and plots distribution of its frequency

    Parameters:
    -----------

    :param data: list of classes
    :param xlabel: name to be shown on x-axis
    :param ylabel: name to be shown on y-axis
    :param image_name: name of image

    Returns:
    --------
    :return: None
    """

    letter_counts = Counter(data)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')

    df.plot(kind='bar', rot=0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig('./' + image_name)

def process_data(xtrain,ytrain,dict):

    """
    this function replace each character of xtrain_obfuscated.txt with unique id (each character has same id)
     and also join each vector with its class label

    Parameters:
    -----------

    :param xtrain: file pointer to xtrain_obsfuscated.txt
    :param ytrain: file pointer to ytrain.txt
    :param dict: dictionary containing unique id of each character

    Returns:
    --------

    :return: array of vectors representing each sentence and corresponding labels
    """

    train_data=[]
    for i in range(len(xtrain)):
        print(i)
        cur_xtrain=[]
        cur_ytrain=[]
        cur_train=[]
        for char in xtrain[i]:
            cur_xtrain.append(dict[char])
        cur_ytrain.append(ytrain[i].replace('\n',''))
        cur_train.append(cur_xtrain)
        cur_train.append(cur_ytrain)

        train_data.append(cur_train)


    training_data=np.array(train_data)

    return training_data

def process_data_test_data(xtest,dict):

    """
    this function replace each character of xtrain_obfuscated.txt with unique id (each character has same id)
     and also join each vector with its class label

    Parameters:
    -----------

    :param xtrain: file pointer to xtest_obsfuscated.txt
    :param dict: dictionary containing unique id of each character

    Returns:
    --------

    :return: array of vectors representing each sentence
    """

    test_data=[]
    for i in range(len(xtest)):
        cur_xtest=[]
        for char in xtest[i]:
            cur_xtest.append(dict[char])
        test_data.append(cur_xtest)

    training_data=np.array(test_data)

    return training_data


def statitistics_train_data(train_data_file):

    """
    this function computes minimum, maximum and average character length of sentences in train data

    Parameters:
    -----------
    :param train_data_file: train data file pointer

    Returns:
    --------

    :return: min, max and avg character length of sentences in train data
    """

    # Min Length
    min_lenth=10000

    for line in train_data_file:
        if len(line)<min_lenth:
            min_lenth=len(line)

    # Max length
    max_len=0
    for line in train_data_file:
        if len(line)>max_len:
            max_len=len(line)

    # Avg length
    length=0
    for line in train_data_file:
        length=length+len(line)

    total_avg=length/len(train_data_file)

    return min_lenth,max_len,total_avg


def main():


    # open class labels text file in read mode
    y_train_file=open('ytrain.txt',mode='r')

    # Read labels and convert it into array of labels
    indices=convert_labels_to_array_of_labels(y_train_file)

    # Plot class distribution
    plot_class_distribution_bar(indices,'Classes','Frequency','class_distribution')


    # Setting parameters for one hot vector
    depth =12
    on_value = 1.0
    off_value = 0.0
    axis = -1

    # Function to convert array of labels to one hot vectors
    one_hot_array=array_to_one_hot_array(indices,depth,on_value,off_value,axis)

    #### create vector of training data and corresponding labels
    # X_train_Data File
    x_train_lines=open('xtrain_obfuscated.txt',mode='r')
    lines_x=x_train_lines.readlines()

    # Y_train_Data File
    y_train_lines=open('ytrain.txt',mode='r')
    lines_y=y_train_lines.readlines()

    # finding unique characters from train data
    txt=''
    for line in lines_x:
        txt += line

    # finding set of characters
    chars = set(txt)

    # create a dictionary of unique character and corresponing ids
    char_indices = dict((c, i+1) for i, c in enumerate(chars))

    # save dictionary
    with open('char_indices.p','wb') as fp:
        pickle.dump(char_indices,fp)

    # function to create array of vectors and corresponding labels of train data
    train_data=process_data(lines_x,lines_y,char_indices)

    # save training data
    with open('training_data.p','wb') as fp:
        pickle.dump(train_data,fp)

    x_test_lines=open('xtest_obfuscated.txt',mode='r')
    lines_x_test=x_test_lines.readlines()

    # function to create array of vectors of test data
    test_data=process_data_test_data(lines_x_test,char_indices)

    # save test data
    with open('training_data.p','wb') as fp:
        pickle.dump(test_data,fp)


    # Statistics of train data at character level
    minimum,maximum,avg_length=statitistics_train_data(lines_x)

if __name__ == '__main__':
    main()


