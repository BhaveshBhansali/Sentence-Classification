import sys
import os
import tensorflow as tf
import numpy as np
from ConvolutionNN import next_x_data,next_y_data,dev_model
import pickle
from random import sample

def main():


    error_out_file=open('./out/LEARNING_RATE_'+str(sys.argv[1]+'_'+'TRAINING_EPOCHS_'+sys.argv[2]+'_'+'N_HIDDEN_'+sys.argv[3]+'_'+'BATCH_SIZE_'+sys.argv[4]+'_dropout_'+sys.argv[5]+'_NumLayers_'+sys.argv[6]+'_Cell_'+sys.argv[7])+'_ML_RNN.txt',mode='a')

    learning_rate=float(sys.argv[1]) # learning rate
    epoch = int(sys.argv[2]) # training epochs
    num_hidden = int(sys.argv[3]) # number of hidden units
    batch_size = int(sys.argv[4]) # batch size
    dropout=float(sys.argv[5]) # dropout probability

    # tf Graph Input
    data = tf.placeholder(tf.int32, [batch_size, 453])
    target = tf.placeholder(tf.float32, [batch_size, 12])

    # embedding generation of each character
    embedding = tf.get_variable("embedding", [78, num_hidden])
    inputs = tf.nn.embedding_lookup(embedding, data)

    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))


    # recurrent cells selection
    if sys.argv[7]=='rnn':
        cell=tf.contrib.rnn.BasicRNNCell(num_hidden,state_is_tuple=True)
    elif sys.argv[7]=='lstm':
        cell=tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
    elif sys.argv[7]=='gru':
        cell=tf.contrib.rnn.GRUCell(num_hidden)


    # number of rnn cells (layers)
    cells=[]
    for _ in range(int(sys.argv[6])):
        cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=dropout,output_keep_prob=dropout)
        cells.append(cell)


    # create multi rnn cell network
    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    val, state = tf.nn.dynamic_rnn(cell,inputs, dtype=tf.float32)

    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    logit=tf.matmul(last, weight) + bias


    out = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logit)

    # Define loss and optimizer
    cost = tf.reduce_mean(out)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # calculate probability for each class
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

    # calculate accuracy
    correct= tf.equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))


    # Y_Train_Data file
    with open('./data/train_data_train.p','rb') as fp:
        train_data=pickle.load(fp)

    with open('./data/train_data_dev.p','rb') as fp:
        train_dev=pickle.load(fp)

    # Create a saver object
    saver = tf.train.Saver()

    no_of_batches = int(len(train_data)/batch_size)

    # character length of each example
    max_length=453

    init_op=tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init_op)

        # Training cycle
        for i in range(epoch):

            prev_index = 0

            # shuffle train data after each epoch
            np.random.shuffle(train_data)

            for j in range(no_of_batches):

                x_train=next_x_data(prev_index,batch_size,train_data,sess,max_length)
                y_train=next_y_data(train_data,prev_index,batch_size,sess)

                prev_index+=batch_size

                sess.run(optimizer,{data: x_train, target: y_train})

            print("Epoch - ",str(i))

            train_dev=sample(list(train_dev),len(train_dev))
            # compute accuracy of model on dev data after each epoch
            dev_model(sess,train_dev,batch_size,max_length,accuracy,prediction,data,target,error_out_file)

        #Now, save the graph
        saver.save(sess, sys.argv[6]+os.sep+str(sys.argv[1]+sys.argv[2]+sys.argv[3]+'convolution_network'))
        print("Optimization Finished!")

    sess.close()


if __name__ == '__main__':
    main()

