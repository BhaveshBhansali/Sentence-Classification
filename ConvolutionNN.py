import tensorflow as tf
import numpy as np
import pickle
from random import sample
import os
import sys
from DataPreprocessing import array_to_one_hot_array


def next_x_data(prev_index,batch_size,data,sess,max_length):

    """
    this function computes batch data

    Parameters:
    -----------

    :param prev_index: starting index of data
    :param batch_size: size of batch
    :param data: data
    :param sess: tensorflow session varibale
    :param max_length: maximum length of each sentence (453 in our case as data has maximum of 453 length of charcters)

    Returns:
    --------
    x:return: data of dimension (batch_size*453(max_length)), each examples of diemension 453
    """

    # Setting parameters for one hot vector
    depth =78  # unique characters in data
    on_value = 1.0
    off_value = 0.0
    axis = -1

    total_indices=[]

    for k in range(prev_index,prev_index+batch_size):
        indices=[]
        for j in data[k][0]:
            indices.append(data[k][0][j])

        if len(indices)>max_length:
            indices=indices[:max_length]
        else:
            indices=indices+[0]*(max_length-len(indices))
        total_indices.append(indices)

    total_indices=np.array(total_indices)/78

    return total_indices
    #cur_line=convert_array_of_labels_to_one_hot_array(total_indices,depth,on_value,off_value,axis)


    #print(sess.run(cur_line).shape)

    #return sess.run(cur_line)


def next_y_data(y_data,prev_index,batch_size,sess):
    """
    this function computes batch data of y labels (one hot vector)
    Parameters:
    -----------

    :param y_data: class labels
    :param prev_index: starting index of data
    :param batch_size: size of batch
    :param sess: tensorflow session varibale

    Returns:
    --------
    :return: one hot vector representation of each class label
    """

    # Setting parameters for one hot vector
    depth =12
    on_value = 1.0
    off_value = 0.0
    axis = -1

    indices=[]
    for i in range(prev_index,prev_index+batch_size):
        indices.append(y_data[i][1][0])

    indices_array=np.array(indices)

    one_hot_array=array_to_one_hot_array(indices_array,depth,on_value,off_value,axis)
    return sess.run(one_hot_array)



def dev_model(sess,train_dev,batch_size,max_length,accuracy,prediction,x,y,file_location):
    """
    compute accuracy of dev data on current state of trained model
    """

    prev_index = 0
    no_of_Batch = int(len(train_dev) / batch_size)

    total_correct = 0.0
    b_count = 0


    #data=sample(data, len(data))

    for i in range(no_of_Batch):
        batch_x = next_x_data(prev_index,batch_size,train_dev,sess,max_length)
        batch_y = next_y_data(train_dev,prev_index,batch_size,sess)

        # Run optimization op (backprop) and cost op (to get loss value)
        correct = sess.run(accuracy,{x: batch_x, y: batch_y})
        #print('Accuracy: '+str(correct))
        b_count += 1
        total_correct += correct

        prev_index+=batch_size


    prediction = total_correct/(1.0*(b_count))

    print("********************************************")
    print("Test Accuracy = "+str(prediction))
    print("********************************************")

    file_location.write("********************************************")
    file_location.write('\n')
    file_location.write("Test Accuracy = "+str(prediction))
    file_location.write('\n')
    file_location.write("********************************************")
    file_location.write('\n')




def conv2d(x, W, b, strides=1):
    """
    this function convolve on x inputs with weights and bias and, non linear activation function (RELU)is applied it

    Parameters:
    -----------

    :param x: input
    :param W: weights
    :param b: bias
    :param strides: how the filter convolves around the input

    Returns:
    --------
    :return: convolved input
    """
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    """
    this function picks important feature over stride (maximum value from stride region)

    Parameters:
    -----------
    :param x: convolved input
    :param k: kernel and stride shape

    Returns:
    --------
    :return: returns important information from convolved input

    """
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


def conv_net_one_conv(x, weights, biases, dropout):
    """
    this function is complete structure of the convolution network

    Parameters:
    -----------
    :param x: input
    :param weights: weights
    :param biases: biases
    :param dropout: dropout probability

    Returns:
    --------
    :return: Output, class prediction values for each class
    """

    x = tf.reshape(x, shape=[-1, 453, 1, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def conv_net_two_conv(x, weights, biases, dropout):

    x = tf.reshape(x, shape=[-1, 453, 1, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



def main():

    error_out_file=open('./out/LEARNING_RATE_'+str(sys.argv[1]+'_'+'TRAINING_EPOCHS_'+sys.argv[2]+'_'+'BATCH_SIZE_'+sys.argv[3])+'CNN_single.txt',mode='a')

    # Parameters
    learning_rate=float(sys.argv[1]) # learning rate
    epoch = int(sys.argv[2])  # training epochs
    batch_size = int(sys.argv[3]) # batch size

    n_input = 453 # data input (each sentence of mximum character length: 453)
    n_classes = 12 # total classes (0-11 classes)
    dropout = 0.75 # keep probability of neurons


    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])



    ##### first convolution model parameters
    weights1 = {
    # 3x3 conv, 1 input, 64 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    # fully connected, batch_size*64*227 inputs, 128 outputs
    'wd1': tf.Variable(tf.random_normal([batch_size*64*227, 128])),
    # 128 inputs, 12 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([128, n_classes]))
    }

    biases1 = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([128])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct first convolution model
    logit = conv_net_one_conv(x, weights1, biases1, dropout)





    ##### second convolution model parameters
    weights2 = {
    # 3x3 conv, 1 input, 64 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    # 3x3 conv, 32 inputs, 128 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    # fully connected, batch_size*128*114 inputs,  256 outputs
    'wd1': tf.Variable(tf.random_normal([batch_size*128*114, 256])), #72960 for batch 10
    # 256 inputs, 12 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([256, n_classes]))
    }

    biases2 = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }


    # Construct second convolution model
    logit = conv_net_two_conv(x, weights2, biases2, dropout)


    # Prediction probability of each class
    prediction=tf.nn.softmax(logit)

    # Output layer with linear activation
    out = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit)

    # Define loss and optimizer
    cost = tf.reduce_mean(out)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    # computing accuracy
    correct= tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))


    # Y_Train_Data file
    with open('./data/train_data_train.p','rb') as fp:
        train_data=pickle.load(fp)

    with open('./data/train_data_dev.p','rb') as fp:
        train_dev=pickle.load(fp)


    # Number of batches
    no_of_batches = int(len(train_data)/batch_size)


    # initialize variables
    init_op=tf.global_variables_initializer()

    # Create a saver object
    saver = tf.train.Saver()

    # character length of each example
    max_length=453

    with tf.Session() as sess:
        sess.run(init_op)

        # Training cycle
        for i in range(epoch):

            prev_index = 0

            # shuffle train data after each epoch
            np.random.shuffle(train_data)

            # Loop over all batches
            for j in range(no_of_batches):
                x_train=next_x_data(prev_index,batch_size,train_data,sess,max_length)
                y_train=next_y_data(train_data,prev_index,batch_size,sess)

                prev_index+=batch_size
                sess.run(optimizer,{x: x_train, y: y_train})

            print("Epoch - ",str(i))

            train_dev=sample(list(train_dev),len(train_dev))
            # compute accuracy of model on dev data after each epoch
            dev_model(sess,train_dev,batch_size,max_length,accuracy,prediction,x,y,error_out_file)

        #Now, save the graph
        saver.save(sess, sys.argv[6]+os.sep+str(sys.argv[1]+sys.argv[2]+sys.argv[3]+'convolution_network'))
        print("Optimization Finished!")

    sess.close()


if __name__ == '__main__':
    main()

